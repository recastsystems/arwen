"""Ported 4.7.1 controls must reach the same solver from public doors."""
from dataclasses import replace
import pytest
from gpuwm.config import RunConfig,validate_run_config
from gpuwm.core.mynn471_gpu import diagnostic_shapes,schema

ROWS=[("bl_mynn_edmf",0),("icloud_bl",0),("tke_budget",1),
      ("bl_mynn_cloudpdf",0),("bl_mynn_cloudpdf",1),
      ("bl_mynn_mixlength",0),("bl_mynn_edmf_mom",0),
      ("bl_mynn_edmf_tke",1),("bl_mynn_cloudmix",0),
      ("bl_mynn_mixqt",1),("bl_mynn_output",1),
      ("bl_mynn_closure",2.5),("bl_mynn_closure",3.0),
      ("bl_mynn_mixscalars",1)]

def config(**kw):
    return RunConfig(nx=8,ny=8,nz=49,dx=3000.,dy=3000.,ztop=20000.,
                     dt=12.,run_seconds=120.,moist=True,mp_physics=8,
                     bl_pbl_physics=5,sf_sfclay_physics=1,**kw)

@pytest.mark.parametrize("key,value",ROWS)
def test_each_new_control_selects_471_and_validates(key,value):
    cfg=config(**{key:value})
    validate_run_config(cfg)
    assert cfg.bl_mynn_version=="wrf_471"
    assert getattr(cfg,key)==value

def test_default_generation_and_optional_outputs_stay_off():
    cfg=config()
    assert cfg.bl_mynn_version=="wrf_461"
    assert diagnostic_shapes(cfg)=={}
    budget=diagnostic_shapes(replace(cfg,tke_budget=1))
    assert set(budget)=={"qshear","qbuoy","qdiss","qwt","dqke"}
    assert budget["dqke"]==(49,8,8)
    assert budget["qwt"]==(50,8,8)

def test_vertical_schema_has_aligned_nonoverlapping_argument_slabs():
    for nz in (20,49,50,64,128):
        meta=schema(nz)
        assert meta["scratch_bytes"]%4==0
        for slab,count in meta["slab_fields"].items():
            ranges=sorted((f["offset"],f["offset"]+f["count"])
                          for f in meta["arguments"].values() if f.get("slab")==slab)
            assert ranges[0][0]==0
            assert ranges[-1][1]==count
            assert all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:]))


def _kernel_arena(nz,lowered=False):
    """Every private-array offset the nz-level kernel text uses, in text
    order, as the kernel computes it (affine expressions evaluated).  With
    ``lowered``, the offsets of model.cu itself: at 49 and 50 levels, the
    two lowerings the fit came from."""
    import re
    from gpuwm.core.mynn471_gpu import DATA,source
    text=(DATA/'model.cu').read_text() if lowered else source(nz)
    found=re.findall(r'm471_array<\w+>\(m471_work,([^;]*?),m471_n\)',text)
    def value(expr):
        expr=expr.replace('MYNN471_NZ',str(nz))
        assert re.fullmatch(r'[\d\s()+*-]+',expr),expr
        return eval(expr,{'__builtins__':{}},{})
    return [value(e) for e in found]


def _slot_shapes(size49,size50):
    """Array shapes (elements per level m and level offset k, or a fixed
    count) whose 4-byte words fill a size49-byte slot at 49 levels and a
    size50-byte slot at 50, with at most 4 bytes of padding, as the two
    lowerings laid them out."""
    def fills(words,size): return 0<=size-4*words<8
    shapes=[('fixed',w) for w in range(size49//4+1) if fills(w,size49) and size50==size49]
    shapes+=[('levels',m,k) for m in range(1,65) for k in (-1,0,1,2)
             if fills(m*(49+k),size49) and fills(m*(50+k),size50)]
    return shapes


@pytest.mark.parametrize("nz",[12,16,24,30,48,49,50,60,65,100,149])
def test_the_private_array_arena_holds_every_array_at_any_depth(nz):
    """The kernel's own arena offsets, fitted through the 49- and 50-level
    lowerings, overlapped below 49 levels, ran into the next array above 50
    and went negative below 24.  Every slot must hold every array shape its
    two lowerings admit, and the arena must fit inside the priced bytes."""
    from gpuwm.core.mynn471_gpu import schema
    at49,at50,atnz=_kernel_arena(49,True),_kernel_arena(50,True),_kernel_arena(nz)
    assert len(at49)==len(at50)==len(atnz)>200
    import json
    from gpuwm.core.mynn471_gpu import DATA
    total49,total50=(json.loads((DATA/f'schema{n}.json').read_text())['scratch_bytes'] for n in (49,50))
    total=schema(nz)['scratch_bytes']
    def slots(offsets,end):
        starts=sorted(set(offsets)|{0})
        size=dict(zip(starts,[b-a for a,b in zip(starts,starts[1:]+[end])]))
        return [size[o] for o in offsets]
    for offset,s49,s50,snz in zip(atnz,slots(at49,total49),slots(at50,total50),slots(atnz,total)):
        assert offset>=0
        shapes=_slot_shapes(s49,s50)
        assert shapes,(s49,s50)
        for shape in shapes:
            words=shape[1] if shape[0]=='fixed' else shape[1]*(nz+shape[2])
            assert 4*words<=snz,(nz,offset,shape,snz)
    if nz==49:
        assert atnz==at49 and total==total49
