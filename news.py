#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import json

# KoBART 라이브러리
from kobart import get_kobart_tokenizer
from transformers import BartForConditionalGenerations

#########################################
# 2. KoBART 토크나이저 전처리 함수
#########################################
def preprocess_for_kobart(input_text, tokenizer, max_length=512):
    """
    KoBART 토크나이저로 문자열을 인코딩해 (input_ids, attention_mask)를 반환하는 함수.
    """
    encoding = tokenizer(
        input_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return encoding


#########################################
# 3. Dataset & DataLoader 구성
#########################################
class NewsSummaryDataset(Dataset):
    """
    (body, title) 형태의 간단한 뉴스 기사 샘플들을
    KoBART 입력형식(input_ids, attention_mask, labels)으로 변환한 Dataset.
    """
    def __init__(self, data_list, tokenizer):
        """
        :param data_list: 리스트 형태, 각 원소가 {'body': 본문, 'title': 제목} 구조
        :param tokenizer: KoBART 토크나이저
        """
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        input_text = sample["body"]
        target_text = sample["title"]

        # 기사 본문(body)을 input으로, 제목(title)을 요약문(target)으로 처리
        input_enc = preprocess_for_kobart(input_text, self.tokenizer, max_length=512)
        target_enc = preprocess_for_kobart(target_text, self.tokenizer, max_length=128)

        # 배치로 묶기 위해 텐서 shape을 [seq_len]로 정돈
        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": target_enc["input_ids"].squeeze(0)
        }

def collate_fn(batch):
    """
    DataLoader에서 batch 형태로 묶을 때 사용되는 collate 함수.
    """
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

#########################################
# 4. KoBART 모델 파인튜닝 & 5. 추론 테스트
#########################################
def main():
    json_file_path = "dataset/sbs_articles.json"  # JSON 파일 경로 (예시)
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    sample_data = [
        {"body": item["normal_body"], "title": item["title"]}
        for item in json_data
        if "normal_body" in item and "title" in item
    ]

    # 2. KoBART 토큰화 준비
    tokenizer = get_kobart_tokenizer()

    # KoBART 사전학습 모델 불러오기 (필요 시 다른 경로/버전 사용)
    model_name = "hyunwoongko/kobart"  # GitHub 등에서 배포되는 KoBART 모델명
    model = BartForConditionalGeneration.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. Dataset & DataLoader
    train_dataset = NewsSummaryDataset(sample_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,      # 예시이므로 배치사이즈 2
        shuffle=True,
        collate_fn=collate_fn
    )

    # 4. KoBART 파인튜닝
    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_epochs = 1   # 실제론 3~5회 이상 반복 권장

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] 평균 Loss: {avg_loss:.4f}")

    # 학습된 모델 저장
    model.save_pretrained("kobart_finetuned")
    tokenizer.save_pretrained("kobart_finetuned")

    # 5. 추론 테스트
    # 저장한 모델 다시 로드(검증 차원)
    finetuned_model = BartForConditionalGeneration.from_pretrained("kobart_finetuned")
    finetuned_model.eval()
    finetuned_model.to(device)

    def generate_summary_kobart(text, max_length=128):
        # 입력 텍스트를 토크나이징
        inputs = tokenizer([text], max_length=512, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = finetuned_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=4,         # 빔 서치 설정
                early_stopping=True
            )
        # 결과 문자열 디코딩
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary

    # 실제 요약/스크립트 생성 테스트 예시
    test_text = (
        "코로나19 사태 이후 각종 산업에서 비대면 서비스가 확산되고 있습니다. "
        "특히 온라인 쇼핑과 원격근무가 보편화되며 전세계 경제 구조가 빠르게 변화하고 있습니다."
    )
    generated_summary = generate_summary_kobart(test_text)
    print("=== 추론 테스트 결과 ===")
    print(generated_summary)


if __name__ == "__main__":
    main()
