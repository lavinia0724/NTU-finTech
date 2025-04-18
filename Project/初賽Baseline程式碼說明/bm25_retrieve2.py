import os
import json
import argparse
import re  # 用於正則表達式過濾非中文字元和數字

from tqdm import tqdm
import jieba  # 用於中文文本分詞
import pdfplumber  # 用於從PDF文件中提取文字的工具
from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data_finance(source_path):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf_finance(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    return corpus_dict

# 讀取單個PDF文件並返回其文本內容，保留中文和數字
def read_pdf_finance(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 輪廓遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            # 保留中文和數字
            chinese_text = keep_chinese_and_numbers(text)
            pdf_text += chinese_text
    pdf.close()  # 關閉PDF文件

    return pdf_text  # 返回萌取出的文本

# 定義函數，保留中文和數字
def keep_chinese_and_numbers(text):
    # 使用正則表達式配符中文字元範圍和數字
    pattern = re.compile(r'[一-鿿0-9]+')
    chinese_chars = pattern.findall(text)
    return ''.join(chinese_chars)

# 根據查詢語句和指定的來源，檢索答案
# 將 TF-IDF 和 BM25 混合使用來提高檢索精度
def BM25_TFIDF_retrieve(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source]

    # 使用 TF-IDF 對文本進行向量化
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_corpus)
    query_vec = vectorizer.transform([qs])

    # 計算相似度分數
    cosine_similarities = np.dot(query_vec, tfidf_matrix.T).toarray()[0]

    # 根據 TF-IDF 分數選擇 top 3 檔案
    top_n = 3
    top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
    top_docs = [filtered_corpus[i] for i in top_indices]

    # 使用 BM25 進行再次檢索
    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in top_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut_for_search(qs))
    ans = bm25.get_top_n(tokenized_query, top_docs, n=1)  # 推薦中最相關的一篇
    best_doc = ans[0]

    # 找回與最佳匹配文本相對應的檔案名
    res = [key for key, value in corpus_dict.items() if value == best_doc]
    return res[0]  # 回傳檔案名




# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    return corpus_dict


# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
    pdf.close()  # 關閉PDF文件

    return pdf_text  # 返回萃取出的文本


# 根據查詢語句和指定的來源，檢索答案
def BM25_retrieve(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source]

    # [TODO] 可自行替換其他檢索方式，以提升效能

    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
    # print(tokenized_corpus)
    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
    a = ans[0]
    # 找回與最佳匹配文本相對應的檔案名
    res = [key for key, value in corpus_dict.items() if value == a]
    return res[0]  # 回傳檔案名



def calculate_accuracy(ground_truth_file, predictions):
    # 讀取 ground truth
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)['ground_truths']

    # 計算正確的預測次數
    correct_predictions = 0
    for gt in ground_truths:
        qid = gt['qid']
        if qid in predictions and predictions[qid] == gt['retrieve']:
            correct_predictions += 1

    # 計算 accuracy
    accuracy = correct_predictions / len(predictions)
    return accuracy

if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
    corpus_dict_insurance = load_data(source_path_insurance)

    source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
    corpus_dict_finance = load_data_finance(source_path_finance)

    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    for q_dict in qs_ref['questions']:
        if q_dict['category'] == 'finance':
            # 進行檢索
            retrieved = BM25_TFIDF_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        # Accuracy: 0.7

        elif q_dict['category'] == 'insurance':
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        # Accuracy: 0.8   

        elif q_dict['category'] == 'faq':
            corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        # Accuracy: 0.9

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

    # 計算並顯示 accuracy
    with open(args.output_path, 'r', encoding='utf8') as f:
        predictions = json.load(f)['answers']
    predictions_map = {pred['qid']: pred['retrieve'] for pred in predictions}

    accuracy = calculate_accuracy('../dataset/preliminary/ground_truths_example.json', predictions_map)
    print(f'Accuracy: {accuracy:.2f}')
