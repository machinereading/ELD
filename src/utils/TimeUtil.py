import time

time_checker = {}
time_millis = lambda: int(round(time.time() * 1000))


def time_analysis(except_keys=[]):

	total_time = sum(time_checker.values())
	time_checker["Time Total"] = total_time
	k = list(sorted([x for x in time_checker.keys() if x not in except_keys], key=lambda x: -time_checker[x]))

	longest = max([len(x) for x in k])
	for item in k:
		v = time_checker[item]
		print(("{0:%d}|{1:10}|{2:10}" % (longest+5)).format(item, "%.2fs" % (v / 1000), "%.2f%%" % (v / total_time * 100)))
	del time_checker["Time Total"]

def add_time_elem(key, amount):
	if key not in time_checker:
		time_checker[key] = 0
	time_checker[key] += amount

def reset_time():
	time_checker = {}

def measure_time(fn):
	def wrapper(*args, **kwargs):
		start_time = time_millis()
		result = fn(*args, **kwargs)
		add_time_elem(fn.__name__, time_millis() - start_time)
		return result
	return wrapper