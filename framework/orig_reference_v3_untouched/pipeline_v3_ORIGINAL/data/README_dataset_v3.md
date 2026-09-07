# ml_training_dataset_v3 — tài liệu

Dataset huấn luyện ML bản **v3 (CSV-driven)**: toàn bộ panel hợp chất, chủng và luật
được nạp từ các file CSV trong `data/inputs/` (không còn hard-code trong Python như v2),
nên mở rộng panel = thêm dòng vào CSV rồi chạy lại, **không sửa code**.

> **Lưu ý trung thực (đọc trước tiên).** Nhãn phản ứng là **mô phỏng** từ expert prior
> (noisy-OR trên các substructure, neo theo EC/PMID), **không** phải kết quả đo thực nghiệm.
> Vì vậy mọi chỉ số đánh giá trên dataset này là **positive control** — đo mức pipeline
> *khôi phục lại chính luật sinh*, **không** phải kết quả sinh học. Con số dự đoán thật chỉ
> có khi nhãn LC-HRMS/MS thật thay nhãn mô phỏng (proposal §1.1.7).

## Cách tạo (tái lập 100%)

```
python3 build_dataset_v3.py
```

Nạp 4 file input (+1 map khối phổ) và sinh bảng huấn luyện. Đường dẫn tự nhận `data/`
hoặc `data1/`, hoặc đặt biến môi trường `SEAGRAMMAR_DATA`.

| Input (`data/inputs/`) | Vai trò |
|---|---|
| `compounds.csv` | 32 hợp chất: `compound_id, compound_name, pillar, class_id, scaffold, smiles, provenance, origin` |
| `strains.csv` | 42 chủng: `strain_id, strain_name, genus, family, phylum, class_tax, role, capabilities, habitat` |
| `reaction_rules.csv` | 15 luật (σ→phản ứng): `reaction, substructure, enzyme, prior_prob, EC, refs` |
| `observations.csv` | Nhãn thực nghiệm thật (đầu vào "during-fellowship"); ghi đè nhãn mô phỏng khi có |
| `ms2_reaction_map.csv` | Bản đồ Δm khối phổ cho suspect screening (`msms_to_observations.py`) |

Đầu ra: `ml_training_dataset_v3.csv` + `DATA_CARD_v3.json`.

## Đặc điểm

- **4.032 dòng** = 32 hợp chất × 42 chủng × 3 replicate sinh học.
- **16 feature** substructure (RDKit/SMARTS từ SMILES) + one-hot `strain_phylum` (+ `habitat`).
- **12 nhãn** multi-label `y_<label>` = 11 phản ứng + `none`
  (`O_demethylation, aldehyde_oxidation, aromatic_hydroxylation, decarboxylation,
  deglycosylation, desulfation, ester_hydrolysis, methylation, reduction, ring_cleavage,
  side_chain_cleavage, none`).
- **12 cột trọng số per-label `w_<label>`** (mới ở v3): trọng số bằng chứng *theo từng nhãn*,
  để một quan sát thực nghiệm của một phản ứng chỉ up-weight đúng nhãn đó — không up-weight
  cả dòng (gồm các nhãn vẫn còn mô phỏng). Có cột `weight` per-row (=max) để tương thích ngược.
- Cột nhóm cho cross-validation: `cv_group` (=`compound_id`, 32 nhóm) cho **leave-compound-out**;
  `class_id` (8 họ) cho **leave-family-out**; `scaffold` (12) cho scaffold-disjoint.

## Cách dùng đúng (QUAN TRỌNG)

- **KHÔNG** dùng làm feature: `source, evidence, strain_id, compound_id, cv_group, provenance,
  smiles, weight, replicate_id`, và **cả 12 cột `w_<label>`** (đây là trọng số, không phải feature).
- **Feature đầu vào:** 16 cột substructure + one-hot `strain_phylum` (+ `habitat`).
- **Chia tập:** `GroupKFold(groups=cv_group)` (leave-compound-out) **và** `groups=class_id`
  (leave-family-out). Báo cáo **song song** cả hai — khoảng cách LCO−LFO định lượng mức rò rỉ
  do scaffold gần trùng.
- **Trọng số:** truyền `w_<label>` vào `sample_weight` (fallback `weight` nếu chưa có).
- **Cổng tối thiểu để đánh giá được:** một nhãn chỉ được đánh giá nếu có ≥ `MIN_POS` (=8) dòng
  dương **và** σ xuất hiện ở ≥ `MIN_COMPOUNDS` (=3 ở mock; mục tiêu 10 ở panel funded) hợp chất
  riêng biệt. Cổng đếm-hợp-chất mới là thứ bảo vệ leave-compound-out — hiện `desulfation` và
  `deglycosylation` (mỗi cái chỉ 2 hợp chất) bị loại **minh bạch**, không báo cáo F1 giả.
- **Metric:** **PR-AUC (average precision) + lift trên prevalence làm chính**, ROC-AUC làm phụ;
  kèm **95% cluster-bootstrap CI** theo nhóm compound. KHÔNG dùng accuracy (baseline toàn-0 ~90%).
  Xem `ml_upgrade/honest_eval.py`.

## Ghi đè bằng dữ liệu thật (`observations.csv`)

Dataset **append-only**: mỗi dòng trong `observations.csv` (một cặp `compound×strain×reaction`
có bằng chứng thật) sẽ ghi đè nhãn mô phỏng tương ứng và nâng trọng số nhãn đó theo evidence
tier (Schymanski). Dòng có `ref` chứa chuỗi placeholder (vd `PMID:XXXX...`) bị **lọc theo thiết
kế** → `DATA_CARD_v3.json.n_experimental_rows_overridden` phản ánh số ghi đè thật (hiện = 0).

## Khác biệt so với v2

| | v2 | v3 |
|---|---|---|
| Nguồn panel | hard-code trong `.py` | **CSV-driven** (`data/inputs/`) |
| Quy mô | 20 × 30 × 3 = 1.800 | **32 × 42 × 3 = 4.032** |
| Trọng số | chỉ `weight` per-row | thêm **`w_<label>` per-label** |
| Ghi đè thực nghiệm | — | qua `observations.csv` |
| Nhóm CV họ/scaffold | — | `class_id`, `scaffold` (hỗ trợ leave-family-out) |

*(README của bản cũ vẫn ở `README_dataset_v2.md` và mô tả đúng dataset v2 1.800 dòng.)*
