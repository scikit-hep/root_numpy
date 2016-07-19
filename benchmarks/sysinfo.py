"""
Get system hardware information
http://stackoverflow.com/a/4194146/1002176
"""
import cpuinfo
import sys, os, fcntl, struct
import pickle

if os.geteuid() > 0:
    print("ERROR: Must be root to use")
    sys.exit(1)

with open(sys.argv[1], "rb") as fd:
    # tediously derived from the monster struct defined in <hdreg.h>
    # see comment at end of file to verify
    hd_driveid_format_str = "@ 10H 20s 3H 8s 40s 2B H 2B H 4B 6H 2B I 36H I Q 152H"
    # Also from <hdreg.h>
    HDIO_GET_IDENTITY = 0x030d
    # How big a buffer do we need?
    sizeof_hd_driveid = struct.calcsize(hd_driveid_format_str)

    # ensure our format string is the correct size
    # 512 is extracted using sizeof(struct hd_id) in the c code
    assert sizeof_hd_driveid == 512

    # Call native function
    buf = fcntl.ioctl(fd, HDIO_GET_IDENTITY, " " * sizeof_hd_driveid)
    fields = struct.unpack(hd_driveid_format_str, buf)
    hdd = fields[15].strip()

cpu = cpuinfo.get_cpu_info()['brand']

print(cpu)
print("Hard Drive Model: {0}".format(hdd))

info = {
    'CPU': cpu,
    'HDD': hdd,
}

with open('hardware.pkl', 'w') as pkl:
    pickle.dump(info, pkl)
