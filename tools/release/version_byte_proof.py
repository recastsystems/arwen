"""Prove the sole producer-stamp edits in a Rust-written NetCDF container.

Build the expected tip bytes from the base grammar, changing only that text,
its lengths, addresses moved by it, alignment padding and HDF checksums. The
input files are never edited. Unknown layouts fail closed.

The release byte rule (R1) allows exactly the global attributes named in
PRODUCER_STAMPS to differ, each arm writing its own true value:

* GPUWM_VERSION, the engine version ArWen history writes (gpuwm/io/wrfout.py);
* git_version, the producer commit the Rust preparation and MPAS
  initial/boundary writers stamp (rw-mpas init/lbc emit.rs).  PREP-LINEAGE-01
  (3.0.0): the 49-boundary prep and hex LBC literal-byte comparisons differed
  only there (base 4123f1916e3 vs candidate), every data byte equal.

Anything else that differs at the byte level is a real red.
"""
from __future__ import annotations
import argparse
import json
import struct
import sys
import numpy as np

_U64_MAX = (1 << 64) - 1

#: The global attributes R1 lets differ between arms, and nothing else.
PRODUCER_STAMPS = ('GPUWM_VERSION', 'git_version')

def _get(b, at, size):
    return int.from_bytes(b[at:at+size], 'little')

def _put(b, at, size, n):
    b[at:at+size] = n.to_bytes(size, 'little')

def _checksum(data):
    # Bob Jenkins lookup3, the same checksum as netcdf-writer/hdf_checksum.rs.
    mask = 0xffffffff
    def rot(x, n): return ((x << n) | (x >> (32-n))) & mask
    a = b = c = (0xdeadbeef + len(data)) & mask
    at = 0
    while at + 12 < len(data):
        a = (a + _get(data, at, 4)) & mask
        b = (b + _get(data, at+4, 4)) & mask
        c = (c + _get(data, at+8, 4)) & mask
        a = ((a-c) & mask) ^ rot(c,4); c = (c+b) & mask
        b = ((b-a) & mask) ^ rot(a,6); a = (a+c) & mask
        c = ((c-b) & mask) ^ rot(b,8); b = (b+a) & mask
        a = ((a-c) & mask) ^ rot(c,16); c = (c+b) & mask
        b = ((b-a) & mask) ^ rot(a,19); a = (a+c) & mask
        c = ((c-b) & mask) ^ rot(b,4); b = (b+a) & mask
        at += 12
    tail = data[at:]
    if tail:
        a = (a + int.from_bytes(tail[:4], 'little')) & mask
        b = (b + int.from_bytes(tail[4:8], 'little')) & mask
        c = (c + int.from_bytes(tail[8:12], 'little')) & mask
        c = ((c ^ b) - rot(b,14)) & mask
        a = ((a ^ c) - rot(c,11)) & mask
        b = ((b ^ a) - rot(a,25)) & mask
        c = ((c ^ b) - rot(b,16)) & mask
        a = ((a ^ c) - rot(c,4)) & mask
        b = ((b ^ a) - rot(a,14)) & mask
        c = ((c ^ b) - rot(b,24)) & mask
    return c

def _object(data, at):
    if data[at:at+6] != b'OHDR\x02\x06':
        raise ValueError('not the Rust compact tracked object-header layout')
    end = at + 10 + _get(data, at+6, 4)
    if _checksum(data[at:end]) != _get(data, end, 4):
        raise ValueError('invalid object-header checksum')
    messages = []
    pos = at + 10
    while pos < end:
        size = _get(data, pos+1, 2)
        messages.append((data[pos], pos, pos+6, pos+6+size))
        pos += 6 + size
    if pos != end:
        raise ValueError('invalid compact message length')
    return end + 4, messages

def _attr(data, begin, end):
    if data[begin:begin+2] != b'\x01\x00':
        raise ValueError('not a Rust attribute-v1 message')
    n, dt, ds = struct.unpack_from('<HHH', data, begin+2)
    name = data[begin+8:begin+8+n].rstrip(b'\0').decode()
    dtype = begin + ((8+n+7) & ~7)
    space = begin + ((dtype-begin+dt+7) & ~7)
    value = begin + ((space-begin+ds+7) & ~7)
    if value > end:
        raise ValueError('invalid attribute extents')
    return name, dtype, space, value

def _hdf_expected(base, version, stamp='GPUWM_VERSION'):
    if base[:16] != b'\x89HDF\r\n\x1a\n' + bytes((0,0,0,0,0,8,8,0)):
        raise ValueError('not the Rust HDF superblock-v0 layout')
    root = _get(base, 64, 8)
    root_end, messages = _object(base, root)
    selected = []
    addresses = []
    for kind, msg, begin, end in messages:
        if kind == 6:
            addresses.append(_get(base, end-8, 8))
        elif kind == 12:
            name, dt, sp, value = _attr(base, begin, end)
            if name == stamp:
                selected.append((msg, begin, end, dt, sp, value))
    if len(selected) != 1 or not addresses or addresses[0] != root_end:
        raise ValueError('missing version or non-contiguous Rust objects')
    msg, begin, end, dt, sp, value = selected[0]
    if base[dt:dt+4] != b'\x13\x01\x00\x00' or base[sp:sp+8] != bytes((1,0,0,0,0,0,0,0)):
        raise ValueError('version must be a scalar fixed text attribute')
    if _get(base, dt+4, 4) != end-value:
        raise ValueError('invalid version datatype size')
    encoded = version.encode('utf-8')
    delta = len(encoded) - (end-value)
    root_blob = bytearray(base[root:root_end])
    root_blob[value-root:end-root] = encoded
    _put(root_blob, dt-root+4, 4, len(encoded))
    _put(root_blob, msg-root+1, 2, end-begin+delta)
    _put(root_blob, 6, 4, _get(base, root+6, 4)+delta)
    edits = [dict(kind=stamp, base_start=value, base_end=end,
                  tip_start=value, tip_end=value+len(encoded)),
             dict(kind='version datatype size', base_start=dt+4, base_end=dt+8),
             dict(kind='version message length', base_start=msg+1, base_end=msg+3),
             dict(kind='root message-chunk length', base_start=root+6, base_end=root+10)]
    # The root links name all following object headers in definition order.
    for kind, _, _, stop in messages:
        if kind == 6:
            p = stop-8
            shifted = p + (delta if p >= end else 0)
            _put(root_blob, shifted-root, 8, _get(base,p,8)+delta)
            edits.append(dict(kind='moved object address',base_start=p,base_end=p+8))
    objects = []
    last = root_end
    for addr in addresses:
        if addr != last:
            raise ValueError('objects are not contiguous')
        last, ms = _object(base, addr)
        objects.append((addr,last,ms))
    old_heap = (last+7) & ~7
    new_heap = (last+delta+7) & ~7
    heap_delta = new_heap-old_heap
    expected = bytearray(base[:root])
    def seal(blob):
        _put(blob,len(blob)-4,4,_checksum(blob[:-4])); return blob
    expected.extend(seal(root_blob))
    for addr, stop, ms in objects:
        blob = bytearray(base[addr:stop])
        for kind, _, start, finish in ms:
            if kind == 12:
                name, _, _, val = _attr(base,start,finish)
                if name == 'DIMENSION_LIST':
                    if (finish-val) % 16:
                        raise ValueError('invalid dimension-list descriptors')
                    for p in range(val+4,finish,16):
                        _put(blob,p-addr,8,_get(base,p,8)+heap_delta)
                        edits.append(dict(kind='moved dimension heap address',base_start=p,base_end=p+8))
        expected.extend(seal(blob))
        edits.append(dict(kind='object checksum',base_start=stop-4,base_end=stop))
    expected.extend(bytes(new_heap-len(expected)))
    pos = old_heap
    while pos < len(base):
        if base[pos:pos+8] != b'GCOL\x01\x00\x00\x00':
            raise ValueError('not a Rust dimension-reference global heap')
        size = _get(base,pos+8,8)
        blob = bytearray(base[pos:pos+size]); p=16
        while _get(blob,p,2):
            n = _get(blob,p+8,8)
            if n != 8 or _get(blob,p+16,8) not in addresses:
                raise ValueError('unexpected heap object; cannot prove a version-only edit')
            _put(blob,p+16,8,_get(blob,p+16,8)+delta)
            edits.append(dict(kind='moved dimension-scale address',base_start=pos+p+16,base_end=pos+p+24))
            p = (p+16+n+7) & ~7
        expected.extend(blob); pos += size
    _put(expected,40,8,len(expected))
    edits.extend([dict(kind='root checksum',base_start=root_end-4,base_end=root_end),
                  dict(kind='root messages relocated after version',base_start=end,
                       base_end=root_end,delta=delta),
                  dict(kind='file EOF address',base_start=40,base_end=48),
                  dict(kind='object relocation',base_start=root_end,base_end=last,delta=delta),
                  dict(kind='heap alignment and relocation',base_start=last,base_end=len(base),delta=heap_delta)])
    return bytes(expected), edits

def _classic_expected(base, version, stamp='GPUWM_VERSION'):
    if base[:4] not in (b'CDF\x01', b'CDF\x02', b'CDF\x05'):
        raise ValueError('not a classic NetCDF container')
    count_size = 8 if base[3] == 5 else 4
    offset_size = 4 if base[3] == 1 else 8
    pos = 4 + count_size
    def number(size):
        nonlocal pos
        if pos+size > len(base):
            raise ValueError('truncated classic header')
        value = int.from_bytes(base[pos:pos+size], 'big'); pos += size
        return value
    def count(): return number(count_size)
    def name():
        nonlocal pos
        n = count(); value = base[pos:pos+n].decode()
        pos += (n+3) & ~3
        return value
    def inventory(tag):
        actual, n = number(4), count()
        if actual not in (0, tag) or (actual == 0 and n):
            raise ValueError('invalid classic inventory')
        return n
    found = []
    def attrs(global_attributes=False):
        nonlocal pos
        sizes = {1:1, 2:1, 3:2, 4:4, 5:4, 6:8, 7:1, 8:2, 9:4, 10:8, 11:8}
        for _ in range(inventory(12)):
            label = name(); dtype = number(4); length_at = pos; n = count()
            if dtype not in sizes:
                raise ValueError('unknown classic attribute type')
            end = pos+n*sizes[dtype]; padded = pos+((n*sizes[dtype]+3) & ~3)
            if global_attributes and label == stamp:
                if dtype != 2:
                    raise ValueError('producer version is not text')
                found.append((length_at,pos,end,padded))
            pos = padded
    for _ in range(inventory(10)):
        name(); count()
    attrs(True)
    offsets = []
    for _ in range(inventory(11)):
        name()
        for _ in range(count()): count()
        attrs(); number(4); count()
        at = pos; value = number(offset_size); offsets.append((at,value))
    if len(found) != 1:
        raise ValueError(f'not a classic header with exactly one {stamp}')
    length_at,start,end,padded = found[0]
    encoded = version.encode('utf-8'); replacement = encoded+bytes((-len(encoded)) % 4)
    delta = len(replacement)-(padded-start)
    # A stamp that keeps its padded length moves nothing, so header free space
    # (MPAS writers reserve some) does not matter.  One that changes it is
    # proven only for the compact Rust layout, where data starts at the header end.
    if delta and (not offsets or min(v for _,v in offsets) != pos):
        raise ValueError('not a compact Rust classic header with one producer version')
    expected = bytearray(base[:start]+replacement+base[padded:])
    expected[length_at:length_at+count_size] = len(encoded).to_bytes(count_size,'big')
    edits = [dict(kind=stamp+' and its alignment',base_start=start,base_end=padded,
                  tip_start=start,tip_end=start+len(replacement)),
             dict(kind='version attribute length',base_start=length_at,base_end=length_at+count_size)]
    for at,value in offsets:
        shifted = at+delta
        expected[shifted:shifted+offset_size] = (value+delta).to_bytes(offset_size,'big')
        edits.append(dict(kind='moved variable data address',base_start=at,base_end=at+offset_size,
                          tip_start=shifted,tip_end=shifted+offset_size))
    if delta:
        edits.append(dict(kind='header and data relocation after version padding',
                          base_start=padded,base_end=len(base),delta=delta))
    return bytes(expected),edits

def _expected(container, stamps):
    """Apply each producer stamp's rewrite in turn; each step is a whole, resealed container.

    Same-length stamps go first, so they move nothing and their edits stay in
    the coordinates the later step reports.  Two length-changing stamps are
    still rebuilt exactly; if an offset then has no single-step explanation
    the comparison fails closed.
    """
    unknown = sorted(set(stamps) - set(PRODUCER_STAMPS))
    if unknown:
        raise ValueError(f'{unknown} are not producer stamps R1 allows')
    if not stamps:
        raise ValueError('no producer stamp to account for the difference')
    edits = []
    order = sorted(stamps, key=lambda name: (name == 'GPUWM_VERSION', name))
    for name in order:
        build = _classic_expected if container.startswith(b'CDF') else _hdf_expected
        container, step = build(container, stamps[name], name)
        edits.extend(step)
    return container, edits

def compare_container(base, tip, tip_version=None, *, stamps=None):
    """Return every raw differing offset, and the exact allowed rewrite proof.

    ``tip_version`` is the tip's GPUWM_VERSION (the original R1 stamp);
    ``stamps`` maps further PRODUCER_STAMPS names present in both files to
    the tip's value (``{'git_version': 'gpuwm-<commit>'}``).
    """
    stamps = dict(stamps or {})
    if tip_version is not None:
        stamps.setdefault('GPUWM_VERSION', tip_version)
    a,b = base.read_bytes(),tip.read_bytes()
    n=min(len(a),len(b))
    offsets=np.flatnonzero(np.frombuffer(a[:n],np.uint8)!=np.frombuffer(b[:n],np.uint8)).tolist()
    offsets.extend(range(n,max(len(a),len(b))))
    result=dict(byte_compare_performed=True,byte_equal=a==b,base_bytes=len(a),tip_bytes=len(b),
                differing_offsets=offsets,version_only=False,allowed_edits=[])
    if a==b:
        result['version_only']=True
        return result
    try:
        expected,edits=_expected(a,stamps)
        result.update(version_only=expected==b,allowed_edits=edits,stamps=sorted(stamps),
                      proof='Exact expected container, using only the declared producer stamps '
                            'and their dependent grammar fields')
        if expected!=b:
            m=min(len(expected),len(b))
            result['unexplained_offsets']=np.flatnonzero(np.frombuffer(expected[:m],np.uint8)!=np.frombuffer(b[:m],np.uint8)).tolist()+list(range(m,max(len(expected),len(b))))
        else:
            groups = {}
            for offset in offsets:
                explanations = [e['kind'] for e in edits if
                    e['base_start'] <= offset < e['base_end'] or
                    e.get('tip_start',e['base_start']+e.get('delta',0)) <= offset <
                    e.get('tip_end',e['base_end']+e.get('delta',0))]
                if not explanations:
                    raise ValueError(f'differing offset {offset} has no grammar explanation')
                groups.setdefault(' / '.join(explanations),[]).append(offset)
            result['differing_offset_groups'] = [dict(reason=k,offsets=v) for k,v in groups.items()]
    except (ValueError,IndexError,OverflowError,struct.error) as error:
        result['version_only']=False
        result['reason']=str(error)
    return result

def stamps_of(base, tip):
    """The PRODUCER_STAMPS present as global attributes in BOTH files, with the tip's values."""
    from netCDF4 import Dataset
    with Dataset(base) as x, Dataset(tip) as y:
        both = set(x.ncattrs()) & set(y.ncattrs())
        return {name: str(y.getncattr(name)) for name in PRODUCER_STAMPS if name in both}

def main(argv=None):
    """Compare file pairs under R1: ``BASE TIP [BASE TIP ...]``; exit 1 on any red."""
    from pathlib import Path
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--json', type=Path, help='write every comparison here')
    args = parser.parse_args(argv)
    if len(args.files) % 2:
        parser.error('give BASE TIP pairs')
    rows = []
    for base, tip in zip(args.files[0::2], args.files[1::2]):
        row = compare_container(base, tip, stamps=stamps_of(base, tip))
        row.update(base=str(base), tip=str(tip))
        rows.append(row)
        print(f"{'PASS' if row['version_only'] else 'RED '} {tip}: "
              f"{len(row['differing_offsets'])} differing bytes"
              + (f", producer stamps only ({', '.join(row.get('stamps', []))})" if row['version_only'] and not row['byte_equal'] else '')
              + (f" -- {row['reason']}" if 'reason' in row else ''))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + '\n')
    return 0 if all(row['version_only'] for row in rows) else 1

if __name__ == '__main__':
    sys.exit(main())
