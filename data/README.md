# RAGEv-Bench Datasets

The RAGEv-Bench datasets are publicly available from:
- **JRC Data Catalogue**: https://data.jrc.ec.europa.eu/dataset/6e8b187d-7bf3-43ae-80a4-f94d5413cfaa
- **European Data Portal**: https://data.europa.eu/data/datasets/6e8b187d-7bf3-43ae-80a4-f94d5413cfaa

## Direct Download Links

| Dataset | Description | Download |
|---------|-------------|----------|
| **ds-VHT.csv** | Virtual Human Twin (VHT) collection — 2,521 Q&A pairs | [FTP](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DIGLIFE/OPEN/RagEv-Bench/ds-VHT.csv) |
| **ds-AMR.csv** | Anti-Microbial Resistance (AMR) collection — 3,656 Q&A pairs | [FTP](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DIGLIFE/OPEN/RagEv-Bench/ds-AMR.csv) |
| **ds-PubMedQA_sample100.csv** | PubMedQA subset — 826 Q&A pairs from 100 documents | [FTP](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DIGLIFE/OPEN/RagEv-Bench/ds-PubMedQA_sample100.csv) |

## Dataset Format

### VHT and AMR collections

| Column | Description |
|--------|-------------|
| `pubid` | Document/publication identifier |
| `question` | Evaluation question |
| `context` | Relevant context from source documents |
| `answer` | Reference answer (ground truth) |

### PubMedQA subset

| Column | Description |
|--------|-------------|
| `pubid` | PubMed article identifier |
| `question` | Evaluation question |
| `context` | Article context (abstracts) |
| `answer` | Reference answer (ground truth) |
| `final_decision` | Binary label (yes/no) for classification evaluation |

## Citation

```
Ceresa, M; Bertolini, L., Comte, V.; Spadaro N.; Raffael, B.; Toussaint, B.;
Consoli, S.; Muñoz Piñeiro A.; Patak, A.; Querci M.; Wiesenthal T. (2026):
RAGEv-Bench. European Commission, Joint Research Centre [Dataset]
doi: 10.2905/JRC.8044N38
PID: http://data.europa.eu/89h/6e8b187d-7bf3-43ae-80a4-f94d5413cfaa
```
