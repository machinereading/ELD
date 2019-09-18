import re

cho = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
jung = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
jong = ['e', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ',
        'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
alpha = [chr(x) for x in range(ord('a'), ord('z') + 1)] + [chr(x) for x in range(ord('A'), ord('Z') + 1)] + [".", ",",
                                                                                                             ":"]
decomposer = {"ㄳ": ["ㄱ", "ㅅ"], "ㄵ": ["ㄴ", "ㅈ"], "ㄶ": ["ㄴ", "ㅎ"], "ㄺ": ["ㄹ", "ㄱ"], "ㄻ": ["ㄹ", "ㅁ"], "ㄼ": ["ㄹ", "ㅂ"],
              "ㅀ": ["ㄹ", "ㅎ"], "ㅄ": ["ㅂ", "ㅅ"]}

eomi = ['은', '는', '이', '가', '을', '를', '의', '이다', '하다', '다', '의', '에', '에서', '으로', '로', '까지', '와', '과', '는지', '는데', '만',
        '도', '부터', '조차']
eogan = ['하였', '했', '에서', '들']
jamo_len = len(cho) + len(jung) + len(jong)

def is_korean_character(char):
	return 0xAC00 <= ord(char) <= 0xD7A3

def is_alphabet(char):
	return 'a' <= ord(char) <= 'z' or 'A' <= ord(char) <= 'Z'

def is_digit(char):
	return '0' <= char <= '9'

def decompose_sent(sentence, decompose=False):
	if type(sentence) is not str:
		print(sentence, "is not string")
	result = []
	for char in sentence:
		i = char_to_jamo(char, decompose=decompose)
		if type(i) is str:
			result.append(i)
		else:
			result += i
	return "".join(result)

def char_to_jamo(character, to_num=False, decompose=False):
	x = ord(character)
	if not is_korean_character(character):
		return character
	x -= 0xAC00
	result = []
	result.append(x % len(jong) if to_num else jong[x % len(jong)])
	x //= len(jong)
	result.append(x % len(jung) if to_num else jung[x % len(jung)])
	x //= len(jung)
	result.append(x % len(cho) if to_num else cho[x % len(cho)])
	result.reverse()
	if decompose:  # 이중받침 분해
		if result[-1] in decomposer:
			result = result[:-1] + decomposer[result[-1]]
	return result

def char_to_elem_ind(character):
	# Not valid char: -1
	# 한글 자모
	# 영어 알파벳(대소문자 구분)
	# 일부 특수문자? (.,:)
	elems = char_to_jamo(character, to_num=True)
	if type(elems) is str:
		# 한국어가 아님
		return [alpha.index(elems) + len(cho) + len(jung) + len(jong) if elems in alpha else len(cho) + len(jung) + len(
			jong) + len(alpha)]
	else:
		cho_ind = elems[0]
		jung_ind = elems[1] + len(cho)
		jong_ind = elems[2] + len(cho) + len(jung)
		return [cho_ind, jung_ind, jong_ind]

def stem_sentence(sentence):
	# print(sentence)
	filter_words = r"[^ ㄱ-ㅎㅏ-ㅣ가-힣a-z-A-Z0-9]+"
	result = []
	for word in re.sub(filter_words, " ", sentence).split():
		eomi_removed = False
		if len(word) == 0: continue
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
		if len(word) > 0:
			result.append(word)
	# print(" ".join(result))
	return result

# def tokenize_sentence(sentence):
# 	def char_mode(char):
# 		if is_korean_character(char):
# 			return 0
# 		if is_alphabet(char):
# 			return 1
# 		if is_digit(char):
# 			return 2
# 		return 3

# 	def split_josa(word):

# 		for e in eomi:
# 			if word.endswith(e):
# 				word = word[:-len(e)]
# 				eomi_removed = True
# 				break
# 		if eomi_removed:
# 			for e in eogan:
# 				if word.endswith(e):
# 					word = word[:-len(e)]
# 					break
# 		if len(word) > 0:
# 			result.append(word)
# 	result = []
# 	# split by non-alphabet
# 	buf = []
# 	mode_buf = -1
# 	char_buf = ""
# 	for word in sentence.split():
# 		for char in word:
# 			cmod = char_mode(char)
# 			if mode_buf != cmod:
# 				if mode_buf != -1:
# 					buf.append(char_buf)
# 				char_buf = ""
# 				mode_buf = cmod
# 			char_buf += char
# 		buf.append(char_buf)
# 		char_buf = ""
# 		mode_buf = -1

# 	buf = list(filter(lambda x: len(x) > 0, buf))

# 	# 현재상태: word단위 split, word 내에서 특수문자/숫자/알파벳 종류 따라 split
# 	# 한국어의 경우 조사 떼는 과정 수행
# 	for item in buf:

# 	return result

def tokenize(sentence):
	result = []
	for token in sentence.replace("\n", " ").split():
		buf = []
		for char in token:
			# korean, non-korean, number, special characters
			if is_korean_character(char):
				buf.append(0)
			elif is_digit(char):
				buf.append(1)
			elif 'a' <= char <= 'z' and 'A' <= char <= 'Z':
				buf.append(2)
			elif re.sub(r"[^ ㄱ-ㅎㅏ-ㅣ가-힣a-z-A-Z0-9]", "", char) == "":
				buf.append(3)
			else:
				buf.append(4)
		tt = []
		word = ""
		last = buf[0]
		buf.append(-1)
		for ind, i in enumerate(buf):
			if i != last or ind == len(buf) - 1:
				# print(word)
				if last == 0:

					# tokenize eomi
					eomi_removed = False
					x = []
					for e in eomi:
						if word.endswith(e):
							x.append(e)
							word = word[:-len(e)]
							eomi_removed = True
							break
					else:
						x = [word]
					if eomi_removed:
						for e in eogan:
							if word.endswith(e):
								x = [word[:-len(e)], e] + x
								break
						else:
							x = [word] + x
					# print("x", x)
					tt += x[:]
				else:
					tt.append(word)
				word = ""
			if i == -1:
				break
			word += token[ind]
			last = i
		result += tt[:]
	return [x for x in result if x != ""]

if __name__ == '__main__':
	print(tokenize("123abc우리(집) 우리집에 왜 왔는지 잘 모르겠다."))
