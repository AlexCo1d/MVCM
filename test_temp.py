'''
only for testing

'''
from model.Former_clsvqa import Former_cls

import re

import numpy as np
# import torch
# from functools import partial
# import torch.nn as nn
# from model.archi import MM
# # 假设的输入维度
# batch_size = 2
# img_size = 448  # 图像大小
# in_chans = 3  # 输入通道数
# num_patches = 196  # 假设patch大小为16
# text_length=100 # 假设文本长度为100
# # 创建模拟输入数据
# fake_images = torch.rand(batch_size, in_chans, img_size, img_size)  # 模拟图像数据
# fake_ids = torch.randint(0, 1000, (batch_size, text_length)).long()  # 模拟文本ids
# fake_labels = torch.randint(0, 2, (batch_size,text_length)).long()  # 模拟标签
# fake_attention_mask = torch.ones(batch_size, text_length)  # 全1的attention mask
# fake_type_ids = torch.zeros(batch_size, text_length).long()  # 假设全部是第一类型的token
#
# # 将模拟数据打包成字典，模拟实际使用中的数据批次
# batch = {
#     "image_1": fake_images,
#     "ids": fake_ids,
#     "labels": fake_labels,
#     "attention_mask": fake_attention_mask,
#     "type_ids": fake_type_ids
# }
#
# # 实例化模型
# model = MM(patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True,local_contrastive_loss=True)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # For a model
# model = model.to(device)
#
# # 将模型转换为评估模式（这对于某些模块如Dropout和BatchNorm很重要）
# model.eval()
#
# # 前向传播
# with torch.no_grad():  # 不计算梯度，减少内存/计算需求
#     output = model(batch)
#
# # 检查输出
# print(output[1].shape)
# print(output[2].shape)

#
# path_to_pth = r'C:\Users\Alex\Downloads\MRM.pth'  # 请替换为你的.pth文件的实际路径
# model_weights = torch.load(path_to_pth,map_location=torch.device('cpu'))
# for key in model_weights['model'].keys():
#     print(key)
#     print(model_weights['model'][key].shape)
#
# model.load_state_dict(model_weights['model'], strict=False)
#
# import os
# import csv
# #
# # Paths to your directories (adjust as necessary)
# base_dir = '/home/data/Jingkai/alex/mimic/files'
# # Path for the output CSV file
# output_csv_path = '/home/data/Jingkai/alex/mimic/training.csv'
#
#
# def find_final_report(content):
#     # Search for the start of the final report
#     start_index = content.find('FINAL REPORT')
#     if start_index != -1:
#         # Return the content from 'FINAL REPORT' onwards
#         return content[start_index:]
#     else:
#         # If 'FINAL REPORT' not found, return None or empty string
#         return None
#
#
# # Open the CSV file for writing
# with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     # Write the header row
#     writer.writerow(['image_path', 'report_content'])
#
#     # Walk through the directory
#     for root, dirs, files in os.walk(base_dir):
#         for file_name in files:
#             if file_name.endswith('.jpg'):
#                 # Construct the full path to the image
#                 image_path = os.path.join(root, file_name)
#
#                 # Change the extension from .jpg to .txt to find the corresponding report
#                 report_filename = os.path.split(image_path)[0] + '.txt'
#                 report_path = report_filename
#                 # Read the report content
#                 try:
#                     with open(report_path, 'r', encoding='utf-8') as report_file:
#                         report_content = report_file.read()
#                         # Find and extract 'FINAL REPORT' content
#                         final_report_content = find_final_report(report_content)
#                         if final_report_content:
#                             # Replace newlines with spaces
#                             final_report_content = final_report_content.replace('\n', ' ').strip()
#                             # Write the image path and processed report content to the CSV
#                             writer.writerow([image_path, final_report_content])
#                         else:
#                             print(f"'FINAL REPORT' not found in: {report_filename}")
#
#                 except FileNotFoundError:
#                     print(f"Report file not found for image: {file_name}")
#
# print("CSV file has been created.")
# import pandas as pd
# df = pd.read_csv(output_csv_path)
# row_count = df.shape[0]
# print(f"CSV 文件的行数为：{row_count}")
# import torch
#
# from VQA.model_VQA import MyVQAModel

# t=torch.load('/home/data/Jingkai/alex/pretrain0/checkpoint-40.pth', map_location='cpu')
# u={}
# u['model']=t['model']
# torch.save(u,'/home/data/Jingkai/alex/weight/MM1.pth')

# from PIL import Image
# import pathlib
# from concurrent.futures import ProcessPoolExecutor
# import time
#
# def resize_image(image_path):
#     """
#     Resize the given image to 448x448, apply grayscale, and measure the time taken.
#     """
#     start_time = time.time()  # 开始计时
#
#     with Image.open(image_path) as img:
#         # 应用RandomResizedCrop等效操作
#         img = img.resize((448, 448), Image.BICUBIC)  # 等效于RandomResizedCrop
#         img = img.convert('L').convert('RGB')  # 等效于Grayscale(num_output_channels=3)
#         img.save(image_path)
#
#     end_time = time.time()  # 结束计时
#     print(f"Processed {image_path.name} in {end_time - start_time:.4f} seconds.")
#
# def main(directory_path):
#     """
#     Recursively traverse the directory, find all JPG images,
#     and resize them in parallel while measuring time.
#     """
#     path = pathlib.Path(directory_path)
#     jpg_images = list(path.glob('**/*.jpg'))
#
#     with ProcessPoolExecutor() as executor:
#         executor.map(resize_image, jpg_images)
#
# main('/mnt/data/yueli/files')

# batch_size = 2
# seq_length = 100
# vocab_size = 1000
# hidden_dim = 768
# # 生成随机的问题和答案的ids和attention mask
# images=torch.rand(batch_size, 3, 448, 448)
# input_ids = torch.randint(low=1, high=vocab_size, size=(batch_size, seq_length))
# attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
# answer_ids = torch.randint(low=1, high=vocab_size, size=(100, seq_length))  # +1是为了bos token
# answer_attention = torch.ones(100, seq_length, dtype=torch.long)
#
# # 由于`rank_answer`函数需要模型的一些内部状态，我们在这里不直接调用它。
# # 下面是如何在一个假设的模型中使用这些输入的示例。
# model=MyVQAModel()
# topk_ids, topk_probs = model(images, input_ids, attention_mask, answer_ids, answer_attention, train=False)
#
# print(f"Topk IDs shape: {topk_ids.shape}, Topk Probs shape: {topk_probs.shape}")
# print   (topk_ids)
# print(topk_probs)
# _, pred_idx = topk_probs[0].max(dim=0)
# i=topk_ids[0][pred_idx]
# print(i)

from model.archi_Former import MM_Former
import torch
from model.archi import MM
from functools import partial
import torch.nn as nn
# fake_images = torch.rand(2, 3, 448, 448)  # 模拟图像数据
# text= ['sadflj123','231']
# batch= {
#     'image1': fake_images,
#     'text': text
#
# }
model = Former_cls(
            img_size=384,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=False,
            num_query_token=32,
            prompt="",
            max_txt_len=256,
            max_output_txt_len=256,
            vit_type="vit_16",
            vit_path="",
            bert='./model/submodule/bert/bert-base-uncased',
            qformer_text_input=True,
            instruct=True,
            distill=True
    )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # For a model
# model = model.to(device)
#
# # 将模型转换为评估模式（这对于某些模块如Dropout和BatchNorm很重要）
# model.eval()
#
# # 前向传播
# with torch.no_grad():  # 不计算梯度，减少内存/计算需求
#     output = model(batch)
#
# # 检查输出
# print(output)
# def extract_sections(report):
#     # This regular expression looks for the sections FINDINGS and IMPRESSION
#     # and extracts all text up to the next all-caps word or the end of the string.
#     pattern = r"(FINDINGS:.*?)(?=\n[A-Z]+:|$)|(IMPRESSION:.*?)(?=\n[A-Z]+:|$)"
#
#     extracted_text = ''
#
#     # Searching the report using the pattern
#     matches = re.findall(pattern, report, re.DOTALL)
#
#     # Each match contains tuples with the content of the sections
#     for match in matches:
#         if match[0].startswith('FINDINGS'):
#             extracted_text += match[0] + ' '
#         elif match[1].startswith('IMPRESSION'):
#             extracted_text += match[1] + ' '
#
#     return extracted_text.strip()
#
import pandas as pd
import csv
import os
import re
#
# # 读取CSV文件
# path='/home/data/Jingkai/alex/mimic'
# meta = pd.read_csv(os.path.join(path,'./mimic-cxr-2.0.0-metadata.csv'), sep=',')
# # train = pd.read_csv(os.path.join(path,'./training.csv'), sep=',')
# folder_path = os.path.join(path,'files')
# with open(os.path.join(path,'./training_mv.csv'), 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(['study_id', 'image_path', 'view_type', 'report_content'])
#     for root, dirs, files in os.walk(folder_path):
#         # files is list of files, root is current full dir.
#         if root.split('/')[-1].startswith('s'):
#             study_id = root.split('/')[-1].replace("s", "")
#             if len(files) > 1:
#                 with open(root + '.txt', 'r') as t:
#                     image_path = []
#                     view_type = []
#                     for filename in files:
#                         dicom_id = filename.split('.')[0]
#                         image_path.append(os.path.join(root, filename))
#                         type=str(meta[meta['dicom_id'] == dicom_id]['ViewPosition'].values[0])
#                         view_type.append(type)
#                     report_content = t.read()
#                     report_content = report_content.replace('\n', ' ')
#                     # 移除多余的空格
#                     report_content = re.sub(r'\s+', ' ', report_content)
#                     report_content = extract_sections(report_content)
#                     image_path = ';'.join(image_path)
#                     view_type = ';'.join(view_type)
#                     writer.writerow([study_id, image_path, view_type, report_content])
#             else:
#                 with open(root + '.txt', 'r') as t:
#                     image_path=files[0]
#                     dicom_id = image_path.split('.')[0]
#                     type = str(meta[meta['dicom_id'] == dicom_id]['ViewPosition'].values[0])
#                     report_content = t.read()
#                     report_content = report_content.replace('\n', ' ')
#                     # 移除多余的空格
#                     report_content = re.sub(r'\s+', ' ', report_content)
#                     report_content = extract_sections(report_content)
#                     writer.writerow([study_id, os.path.join(root, image_path), type, report_content])
#
#
#
#
# df = pd.read_csv(os.path.join(path,'./training_mv.csv'), sep=',')
# df = df[df['report_content'].notna()]
# df.to_csv(os.path.join(path,'./training_mv.csv'), index=False)
#
# from model.submodule.bert.xbert import BertLMHeadModel
# tokenizer_config='../model/submodule/bert/bert-base-uncased'
# text_decoder = BertLMHeadModel.from_pretrained(tokenizer_config)
# import torch
# from transformers import BertTokenizer
# from model.submodule.bert.xbert import BertLMHeadModel
#
# # 配置和初始化模型和tokenizer
# tokenizer_config = '../model/submodule/bert/bert-base-uncased'
# text_decoder = BertLMHeadModel.from_pretrained(tokenizer_config)
# tokenizer = BertTokenizer.from_pretrained(tokenizer_config)
#
# # 随机生成输入数据
# batch_size = 2
# seq_length = 16
# hidden_size = text_decoder.config.hidden_size
#
# # 随机生成输入id和attention mask
# input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length)).to(torch.int64)
# attention_mask = torch.randint(0, 2, (batch_size, seq_length)).to(torch.int64)
#
# # 随机生成encoder hidden states
# encoder_hidden_states = torch.randn(batch_size, 32, hidden_size)
# device = torch.device('cpu')
# # 随机生成encoder attention mask
# encoder_attention_mask = torch.ones((batch_size, 32)).to(torch.int64)
# query_atts = torch.ones(encoder_hidden_states.size()[:-1], dtype=torch.long).to(device)
# # 设置模型设备
#
# text_decoder.to(device)
# input_ids = input_ids.to(device)
# attention_mask = attention_mask.to(device)
# encoder_hidden_states = encoder_hidden_states.to(device)
# encoder_attention_mask = encoder_attention_mask.to(device)
#
# # 前向传播测试
# outputs = text_decoder(
#     input_ids,
#     attention_mask=attention_mask,
#     encoder_hidden_states=encoder_hidden_states,
#     encoder_attention_mask=encoder_attention_mask,
#     labels=input_ids,
#     return_dict=True,
#     reduction='none'
# )
# print(outputs.loss)
# import json
# import os
# import pandas as pd
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge import Rouge
# import pymeteor.pymeteor as pymeteor
# from Generation.Dataset import pre_caption
#
# data=pd.read_json(r'C:\Users\Alex\Desktop\gen_result_iu (1).json')
# bleu_scores = []
# rouge_scores = []
# meteor_scores = []
# rouge = Rouge()  # Initialize the Rouge metric
# for i, t in enumerate(data.iterrows()):
#     print(i)
#     item=data.iloc[i]
#     gt= pre_caption(item['gt'])
#     gen = pre_caption(str(item['gen']))
#     weights = [(1.0, 0, 0, 0),  # BLEU-1
#                    (0.5, 0.5, 0, 0),  # BLEU-2
#                    (0.33, 0.33, 0.33, 0),  # BLEU-3
#                    (0.25, 0.25, 0.25, 0.25)]  # BLEU-4
#     bleu_score = [sentence_bleu([gt], gen, weights=w) for w in weights]
#     bleu_scores.append(bleu_score)
#     print(bleu_score)
#     # Compute ROUGE score
#     rouge_score = rouge.get_scores(gen, gt)[0]['rouge-l']['f']  # get_scores returns a list of scores per item
#     rouge_scores.append(rouge_score)
#     print(rouge_score)
#
#     # Compute METEOR score
#     meteor_score = pymeteor.meteor(gt, gen)
#     meteor_scores.append(meteor_score)
#     print(meteor_score)
#     # gt=tokenizer.tokenize({0: [{'image_id': 0, 'caption': gt.encode('utf-8')}]})
#     # gen=tokenizer.tokenize({0: [{'image_id': 0, 'caption': gen.encode('utf-8')}]})
#     # cider_score, _ = cider.compute_score(gt, gen)
#     # print(f"Image {i} CIDEr Score: {cider_score}")
#
# # Here you can compute average scores or handle results as needed
# avg_bleu_1 = sum([i[0] for i in bleu_scores] ) / len(bleu_scores)
# avg_bleu_2 = sum([i[1] for i in bleu_scores] ) / len(bleu_scores)
# avg_bleu_3 = sum([i[2] for i in bleu_scores] ) / len(bleu_scores)
# avg_bleu_4 = sum([i[3] for i in bleu_scores] ) / len(bleu_scores)
# avg_rouge = sum(rouge_scores) / len(rouge_scores)
# avg_meteor = sum(meteor_scores) / len(meteor_scores)
#
# # Print or return the computed metrics
# print(f"Average BLEU-1 Score: {avg_bleu_1}")
# print(f"Average BLEU-2 Score: {avg_bleu_2}")
# print(f"Average BLEU-3 Score: {avg_bleu_3}")
# print(f"Average BLEU-4 Score: {avg_bleu_4}")
# print(f"Average ROUGE Scores: {avg_rouge}")
# print(f"Average METEOR Score: {avg_meteor}")

# from cidereval import cider, ciderD
# gt=tokenizer.tokenize({0: [{'image_id': 0, 'caption': gt.encode('utf-8')}]})
# gen=tokenizer.tokenize({0: [{'image_id': 0, 'caption': gen.encode('utf-8')}]})
# cider_score, _ = cider.compute_score(gt, gen)
# print(f"Image {i} CIDEr Score: {cider_score}")


# jupyter remote command, then connect:
# jupyter notebook --no-browser --allow-root

# ROCO!
# export CUDA_VISIBLE_DEVICES=0,5;
# export OMP_NUM_THREADS=1;
# python -m torch.distributed.launch --nnodes=1 --master_port 23226 --nproc_per_node=2 --use_env main_pretrain.py \
#     --num_workers 6 \
#     --mv \
#     --accum_iter 1 \
#     --batch_size 32 \
#     --epochs 10 \
#     --warmup_epochs 1 \
#     --img_size 224 \
#     --scale 1 \
#     --blr 2e-4 --weight_decay 0.02 \
#     --min_lr 4e-6 \
#     --distill_model --vit_type vit_16 \
#     --vit_path /home/data/Jingkai/alex/weight/deit_base_patch16_224.pth \
#     --resume /home/data/Jingkai/alex/weight/MMFormer_60-all.pth --start_epoch 0 \
#     --data_path /home/data/Jingkai/alex/mimic \
#     --output_dir /home/data/Jingkai/alex/pretrain_distill_x