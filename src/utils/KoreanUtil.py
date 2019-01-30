import re
cho = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
jung = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
jong = ['e','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
decomposer = {"ㄳ": ["ㄱ","ㅅ"], "ㄵ": ["ㄴ","ㅈ"], "ㄶ": ["ㄴ", "ㅎ"], "ㄺ": ["ㄹ", "ㄱ"], "ㄻ": ["ㄹ", "ㅁ"], "ㄼ": ["ㄹ", "ㅂ"], "ㅀ": ["ㄹ","ㅎ"], "ㅄ": ["ㅂ","ㅅ"]}

eomi = ['은', '는', '이', '가', '을', '를', '의', '이다', '하다', '다', '의', '에', '에서', '으로', '로', '까지', '와', '과']
eogan = ['하였', '했', '에서', '들']
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

def stem_sentence(sentence):
	# print(sentence)
	filter_words = r"[^ ㄱ-ㅎㅏ-ㅣ가-힣a-z-A-Z0-9]+"
	result = []
	for word in re.sub(filter_words, "", sentence).split():
		eomi_removed = False
		for e in eomi:
			if word.endswith(e):
				word = word[:-len(e)]
				eomi_removed = True
				break
		if eomi_removed:
			for e in eogan:
				if word.endswith(e):
					word = word[:-len(e)]
					break

		result.append(word)
	# print(" ".join(result))
	return result