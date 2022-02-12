import h5py

f = h5py.File('door-v0_demos_clipped.hdf5', 'r')
a = f['actions']
print(a.shape, a.dtype)
print(a[0])