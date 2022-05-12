import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

data_dir='../data/shulex/'

dir_path = data_dir+'bert-pair/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)



def get_category(jsonObj):
    # map the tag list into single lines
    df = jsonObj.drop('label_tag', axis=1).join(
        jsonObj['label_tag'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('tag'))
    # write the tag list
    cate_ls = df['tag'].unique()
    return cate_ls

input_file = '../../shulex/shulex_data.jsonl'


def convert_label(x, only_tag=False):
    res = []
    for i in x:
        if only_tag:
            res.append(i['tag'])
        else:
            res.append((i['span'], i['tag']))
    return res


# load
jsonObj = pd.read_json(path_or_buf=input_file, lines=True)
# convert label to (sentence, tag) list
jsonObj['label_conv'] = jsonObj['label'].map(lambda x: convert_label(x))
jsonObj['content'] = jsonObj['content'].map(lambda x: x.replace('\n', ''))
jsonObj['content'] = jsonObj['content'].map(lambda x: x.replace('\t', ' '))
jsonObj['label_tag'] = jsonObj['label'].map(lambda x: ','.join(convert_label(x, only_tag=True)))
jsonObj['idx'] = list(jsonObj.index)

# train test split
x_train, x_test = train_test_split(jsonObj, test_size=0.2, random_state=42)
print ("train sample: ",x_train.iloc[0,:]['content'])
print ("test sample: ",x_test.iloc[0,:]['content'])
cate_ls = get_category(jsonObj)

print ("category list: ",cate_ls)

with open(dir_path+"test_NLI_M.csv","w",encoding="utf-8") as g:
    for i in tqdm(range(len(x_test))):
       category = []
       id = str(x_test.iloc[i, :]['idx'])
       text = x_test.iloc[i, :]['content']
       label = x_test.iloc[i, :]['label']
       for j in range(len(label)):
           category.append(label[j]['tag'])
           for cate in cate_ls:
               if cate in category:
                   # qa-m
                   g.write(
                       id + "\t" + "yes" + "\t" + "what do you think of the " + cate + " of it ?" + " " + text + "\n")
                   # print (id + "\t" + "yes" + "\t" + cate + "\t" + text + "\n")
               else:
                   # qa-m
                   g.write(
                       id + "\t" + "none" + "\t" + "what do you think of the " + cate + " of it ?" + " " + text + "\n")
                   # print(id + "\t" + "none" + "\t" + cate + "\t" + text + "\n")



with open(dir_path+"train_NLI_M.csv","w",encoding="utf-8") as g:
     for i in tqdm(range(len(x_train))):
        category = []
        id = str(x_train.iloc[i, :]['idx'])
        text = x_train.iloc[i, :]['content']
        label = x_train.iloc[i, :]['label']

        for j in range(len(label)):
            category.append(label[j]['tag'])
            for cate in cate_ls:
                if cate in category:
                    # qa-m
                    g.write(
                        id + "\t" + "yes" + "\t" + "what do you think of the " + cate + " of it ?" + " " + text + "\n")
                    # print (id + "\t" + "yes" + "\t" + cate + "\t" + text + "\n")
                else:
                    # qa-m
                    g.write(
                        id + "\t" + "none" + "\t" + "what do you think of the " + cate + " of it ?" + " " + text + "\n")
                    # print(id + "\t" + "none" + "\t" + cate + "\t" + text + "\n")
