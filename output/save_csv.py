import csv
import codecs

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
            file_csv = codecs.open(file_name,'w+','utf-8')#追加
            writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            # for data in datas:
            #     writer.writerow(data)
            writer.writerows(map(lambda x: [x], datas))
            print("Written done!")