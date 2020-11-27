import pickle
import re

PATH_FILES = ["data/luc-van-tien.txt"]
START_OF_SENTENCE = "{"
END_OF_SENTENCE = "}"
MAX_VIETNAMESE_LENGTH = 22
NUMBER_OF_CHARACTER_TO_TRAIN = 5
REGEX_PUNCTUATIONS = "[^{}a-zA-ZạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]"

output = set()
for path_file in PATH_FILES:
    with open(path_file, "r", encoding='utf-8') as file:
        for line in file:
            line = line.lower()
            line = re.sub(REGEX_PUNCTUATIONS, " ", line)
            sentence = " ".join([w for w in line.split(" ") if w != ''])
            sentence = "{}{}{}".format(START_OF_SENTENCE, sentence, END_OF_SENTENCE)
            output.add(sentence)
total_sentences = " ".join(sorted(output))
total_characters = "".join(set(total_sentences))
total_characters = sorted(total_characters)
total_characters_mapping = dict((c, i) for i, c in enumerate(total_characters))
pickle.dump(total_characters_mapping, open('data/nlp_character_vocab.pkl', 'wb'))

final_output = []
for i in range(0, len(total_sentences) - MAX_VIETNAMESE_LENGTH):
    final_output.append(total_sentences[i : i + MAX_VIETNAMESE_LENGTH] + '\n')

with open("data/nlp_dataset.txt", 'w') as file:
    file.writelines(line for line in final_output)