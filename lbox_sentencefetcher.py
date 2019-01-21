import pymysql
conn = pymysql.connect(host='kbox.kaist.ac.kr', port=3142, user='root', password='swrcswrc',
						   db='KoreanWordNet2', charset='utf8', autocommit=True)
curs=conn.cursor()
debug_str = ""
def get_document_name_from_id(sent_id):
	
	SQL = "select convert(sentence using utf8) from Wiki_Sentence_TBL_20180801 where global_s_id=%s" % sent_id
	curs.execute(SQL)
	row = curs.fetchall()
	try:
		return row[0][0]
	except Exception:
		print(sent_id)
		return ""


if __name__ == '__main__':
	with open("/home/minho/ref/mulrel-nel/error_log.txt", encoding="UTF8") as f, open("errors.txt", "w", encoding="UTF8") as wf:
		for line in f.readlines():
			wf.write(get_document_name_from_id(int(line.strip().split("\t")[-1]))+"\n")