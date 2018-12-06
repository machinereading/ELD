def progress(curprog, total, size=10):
	if curprog < 0 or total < 0 or size < 0:
		return ""
	a = int(curprog/total*size)
	return "["+("*"*a)+(" "*(size - a)) + "]"

def printfunc(s):
	print("\r"+s, end="", flush=True)