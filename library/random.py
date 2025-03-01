###! no imports

# split key n times with quadratic probe
# type: (int, int) -> List[int]
def split_key(key, n=2, mod=1e9):
	if key==0: print('Warning: using zero key')
	keys = [int((i*key**2) % mod) for i in range(1,1+n)]
	return keys
