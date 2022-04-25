import os
from vncorenlp import VnCoreNLP

raw_data_path = r"data\raw"
processed_data_path = r"data\processed"
modes = ["train", "dev", "test"]

annotator = VnCoreNLP("VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')

def process_data(mode):
    file_path = os.path.join(raw_data_path, mode)
    for file in os.listdir(file_path):
        if file.split(".")[-1] == "in":
            with open(os.path.join(file_path, file), "r", encoding="utf-8") as fr, open (os.path.join(file_path, "seq_processed.in"), "w", encoding="utf-8") as fw:
                texts = fr.readlines()
                for text in texts:
                    text_segments = annotator.tokenize(text)
                    text_segments = ' '.join(text_segments[0])
                    
                    fw.write(text_segments + "\n")
                    
            # with open(os.path.join(file_path, file), "r", encoding="utf-8") as f:
                

if __name__ == "__main__":
    for mode in modes:
        process_data(mode)       
# with open(r"data\raw\train\seq.in", "r", encoding="utf-8") as f:
#     texts = f.readlines()
#     processed_text = []
#     for text in texts:
#         text_segments = annotator.tokenize(text)
#         for t in text_segments:
#             processed_text.append(' '.join(t))
    
#     print(processed_text)
    
    

text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."


# To perform word segmentation only
word_segmented_text = annotator.tokenize(text) 
print(word_segmented_text)