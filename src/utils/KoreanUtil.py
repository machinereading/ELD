cho = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
jung = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
jong = ['e','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
decomposer = {"ㄳ": ["ㄱ","ㅅ"], "ㄵ": ["ㄴ","ㅈ"], "ㄶ": ["ㄴ", "ㅎ"], "ㄺ": ["ㄹ", "ㄱ"], "ㄻ": ["ㄹ", "ㅁ"], "ㄼ": ["ㄹ", "ㅂ"], "ㅀ": ["ㄹ","ㅎ"], "ㅄ": ["ㅂ","ㅅ"]}
jamo_len = len(cho) + len(jung) + len(jong)

def is_korean_character(char):
	return  0xAC00 <= ord(char) <= 0xD7A3
def is_digit(char):
	return '0' <= char <= '9'
def decompose_sent(sentence, decompose=False):
	if type(sentence) is not str:
		print(sentence, "is not string")
	result = []
	for char in sentence:
		i = char_to_elem(char, decompose=decompose)
		if type(i) is str:
			result.append(i)
		else:
			result += i
	return "".join(result)

def char_to_elem(character, to_num=False, decompose=False):
	x = ord(character)
	if not is_korean_character(character): return character
	x -= 0xAC00
	result = []
	result.append(x%len(jong) if to_num else jong[x % len(jong)])
	x //= len(jong)
	result.append(x%len(jung) if to_num else jung[x % len(jung)])
	x //= len(jung)
	result.append(x%len(cho) if to_num else cho[x % len(cho)])
	result.reverse()
	if decompose: # 이중받침 분해
		if result[-1] in decomposer:
			result = result[:-1]+decomposer[result[-1]]
	return result

