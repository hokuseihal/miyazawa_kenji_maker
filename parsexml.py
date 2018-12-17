def parsetag(taglist=['price','name']):
    import re

    flag_in = False
    texts_with_tag = {tag:[] for tag in taglist}
    text={tag:'' for tag in taglist}

    with open("sample.xml") as f:
        line = f.readline()
        while line:
            for tag in taglist:

                if '<' + tag + '>' in line and '</' + tag + '>' in line:
                    texts_with_tag[tag].append(re.sub(r'<.*?>', '', line))

                elif '</' + tag + '>' in line:
                    flag_in = False
                    texts_with_tag[tag].append(text)

                elif '<' + tag + '>' in line:
                    text[tag] = re.sub(r'<.*?>', '', line)
                    flag_in = True
                elif flag_in:
                    text[tag] += line
            line = f.readline()

    return texts_with_tag


if __name__ == "__main__":
    print(parsetag())
print("END")
