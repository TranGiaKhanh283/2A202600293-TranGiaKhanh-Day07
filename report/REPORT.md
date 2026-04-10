# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trần Gia Khánh
**Nhóm:** B5
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector gần cùng hướng, tức là hai câu/đoạn có nội dung ngữ nghĩa tương đồng. Trong retrieval, điều này thường cho thấy chunk được trả về có khả năng liên quan trực tiếp tới query.

**Ví dụ HIGH similarity:**
- Sentence A: "Machine learning models learn patterns from data."
- Sentence B: "ML systems improve by finding patterns in training data."
- Tại sao tương đồng: Cùng nói về mô hình ML học từ dữ liệu và nhận diện pattern.

**Ví dụ LOW similarity:**
- Sentence A: "Vector databases store embeddings for semantic search."
- Sentence B: "Baking bread requires flour, water, and yeast."
- Tại sao khác: Hai câu thuộc hai domain không liên quan (AI systems vs cooking).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào hướng của vector nên phản ánh tốt mức tương đồng ngữ nghĩa, ít bị ảnh hưởng bởi độ lớn vector. Với text embeddings chiều cao, Euclidean distance thường kém ổn định hơn khi các điểm có xu hướng gần nhau về khoảng cách tuyệt đối.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap)) = ceil((10000 - 50)/(500 - 50)) = ceil(9950/450) = ceil(22.11...)`
> *Đáp án:* `23 chunks`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Nếu overlap = 100 thì `num_chunks = ceil((10000 - 100)/(500 - 100)) = ceil(9900/400) = 25`, tức tăng từ 23 lên 25 chunks. Overlap nhiều hơn giúp giữ ngữ cảnh ở biên giữa các chunk, giảm nguy cơ mất ý khi truy xuất.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** AI Foundations from Wikipedia (AI, ML, DL, Embeddings, LLMs)

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain AI foundations vì bám sát nội dung lab (embedding, chunking, retrieval), nên dễ thiết kế benchmark queries có thể kiểm chứng bằng nguồn rõ ràng. Wikipedia có cấu trúc section tốt, nhiều định nghĩa chuẩn, giúp so sánh chiến lược chunking công bằng hơn. Domain này cũng phù hợp để đối chiếu giữa strategy truyền thống và strategy agentic của thành viên khác.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Artificial-intelligence-Wikipedia.md | https://en.wikipedia.org/wiki/Artificial_intelligence | 84236 | source, extension, language, category, doc_id |
| 2 | Deep-learning-Wikipedia.md | https://en.wikipedia.org/wiki/Deep_learning | 55353 | source, extension, language, category, doc_id |
| 3 | Embedding-machine-learning-Wikipedia.md | https://en.wikipedia.org/wiki/Embedding_(machine_learning) | 2079 | source, extension, language, category, doc_id |
| 4 | Large-language-model-Wikipedia.md | https://en.wikipedia.org/wiki/Large_language_model | 57118 | source, extension, language, category, doc_id |
| 5 | Machine-learning-Wikipedia.md | https://en.wikipedia.org/wiki/Machine_learning | 59619 | source, extension, language, category, doc_id |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string | `Deep-learning-Wikipedia` | Cho phép filter đúng tài liệu khi benchmark (tránh lẫn giữa các chủ đề gần nhau). |
| `language` | string | `en` | Hữu ích khi tập dữ liệu đa ngôn ngữ; giảm nhiễu theo ngôn ngữ truy vấn. |
| `source` | string | `https://en.wikipedia.org/wiki/Machine_learning` | Truy vết nguồn khi đánh giá grounding và ghi báo cáo. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Artificial-intelligence-Wikipedia.md | FixedSizeChunker (`fixed_size`) | 234 | 399.8 | Trung bình, dễ cắt giữa ý |
| Artificial-intelligence-Wikipedia.md | SentenceChunker (`by_sentences`) | 191 | 439.8 | Khá tốt, giữ câu nhưng chunk dài |
| Artificial-intelligence-Wikipedia.md | RecursiveChunker (`recursive`) | 339 | 246.9 | Tốt, chunk ngắn và coherent hơn |

### Strategy Của Tôi

**Loại:** RecursiveChunker (`recursive_400`)

**Mô tả cách hoạt động:**
> Strategy `recursive_400` tách văn bản theo thứ tự separator ưu tiên (đoạn -> dòng -> câu -> từ), chỉ tách mạnh hơn khi phần hiện tại vẫn vượt `chunk_size=400`. Cách này giúp tránh việc cắt mù theo ký tự cố định và giữ ngữ cảnh tốt hơn fixed-size trong tài liệu dài. Với Wikipedia (nhiều section và câu dài), recursive thường tạo chunk gọn hơn nhưng vẫn bám cụm ý. Đây là baseline mạnh để so sánh trực diện với strategy agentic chunking.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Bộ Wikipedia có cấu trúc bán chuẩn: heading, đoạn văn, câu dài/ngắn đan xen. Recursive tận dụng cấu trúc này tốt hơn fixed chunk vì ưu tiên tách ở biên ngữ nghĩa trước khi fallback theo kích thước. Vì vậy strategy này phù hợp để làm mốc so sánh với approach agentic (thông minh hơn nhưng phức tạp hơn).

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Wikipedia domain (5 docs) | best baseline (`by_sentences`, chunk_size~400) | ít chunk hơn recursive | dài hơn recursive | Top-1 dễ lệch ý với query định nghĩa ngắn |
| Wikipedia domain (5 docs) | **của tôi** (`recursive_400`) | nhiều chunk hơn | ngắn hơn, tập trung ý hơn | Kết quả ổn định hơn cho query định nghĩa/khái niệm |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | `recursive_400` | 6/10 | Dễ triển khai, semantic chunking tốt hơn fixed | Một số top-1 vẫn lệch trọng tâm khi bài quá dài |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Ở thời điểm hiện tại, `recursive_400` là baseline tốt để chạy ổn định và minh bạch cho so sánh nhóm. Tuy nhiên, với tài liệu Wikipedia dài và nhiều mục phụ, strategy agentic có tiềm năng vượt trội nếu biết gom chunk theo intent truy vấn thay vì chỉ dựa vào separator. Nhóm nên chốt bằng benchmark chung: so tỉ lệ top-3 relevant và chất lượng answer grounding.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Mình dùng regex tách câu theo dấu kết thúc phổ biến (`.`, `!`, `?`) kết hợp khoảng trắng/newline phía sau để giữ dấu câu trong sentence. Sau đó gom mỗi `max_sentences_per_chunk` câu thành một chunk và strip khoảng trắng dư. Edge cases gồm input rỗng, nhiều khoảng trắng liên tiếp, hoặc text không tách được câu rõ ràng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Recursive chunking thử tách theo thứ tự separator ưu tiên (`\n\n`, `\n`, `. `, ` `, rồi fallback ký tự). Nếu đoạn hiện tại nhỏ hơn `chunk_size` thì dừng (base case) và trả ra chunk đó. Nếu vẫn quá lớn sau một separator, hàm đệ quy gọi tiếp với separator kế tiếp cho đến khi kiểm soát được độ dài.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` embed từng document/chunk rồi lưu record gồm `content`, `embedding`, `metadata` (in-memory; Chroma là tùy chọn). `search` embed query và tính score theo dot product/cosine-like ranking trên toàn bộ records, sau đó sort giảm dần và lấy top-k. Cách này đảm bảo API đơn giản nhưng vẫn đủ để benchmark retrieval.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` lọc metadata trước để thu hẹp candidate set, rồi mới chạy similarity search để tăng precision. `delete_document` xóa theo `doc_id` trong metadata để remove toàn bộ chunks của tài liệu đó. Nếu backend Chroma không khả dụng, hệ thống fallback về in-memory với logic tương đương.

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent chạy theo RAG pattern: retrieve top-k chunks, ghép thành context blocks, rồi đưa vào prompt. Prompt yêu cầu model chỉ trả lời dựa trên context và nói "không biết" khi thiếu bằng chứng. Cấu trúc này giúp giảm hallucination và tăng khả năng kiểm tra grounding.

### Test Results

```
# pytest tests/ -q
# ..........................................                               [100%]
# 42 passed in 0.06s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Machine learning models learn from data. | Models in machine learning improve using data. | high | -0.0336 | Không |
| 2 | Vector stores retrieve semantically similar chunks. | Semantic search uses vector databases to find relevant text. | high | 0.2056 | Đúng |
| 3 | Deep learning uses neural networks. | Baking bread requires flour and yeast. | low | -0.0104 | Đúng |
| 4 | Python is often used for AI prototyping. | Python helps teams build AI prototypes quickly. | high | 0.2153 | Đúng |
| 5 | Metadata filters improve retrieval precision. | Cosine similarity compares the angle between vectors. | low | 0.0255 | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là cặp (1) có nghĩa rất gần nhau nhưng score lại âm nhẹ. Điều này cho thấy với mock embedder deterministic, điểm similarity không phản ánh ngữ nghĩa tốt như embedding model thật. Vì vậy phần này hữu ích để hiểu công thức cosine, còn benchmark chất lượng retrieval thực tế nên dùng local/openai embeddings.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What is artificial intelligence and what capabilities are usually associated with it? | AI is the capability of computational systems to perform tasks linked to human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. |
| 2 | In deep learning, what does the word 'deep' refer to? | It refers to using multiple layers in neural networks; more precisely, the depth of transformations from input to output. |
| 3 | What is an embedding in machine learning and why is cosine similarity commonly used with embeddings? | An embedding maps high-dimensional/complex data into lower-dimensional vectors that preserve relationships; cosine similarity compares direction and is less biased by magnitude/frequency. |
| 4 | What are large language models typically trained to do? | Large language models are trained on large text corpora to predict and generate language, supporting tasks like completion, question answering, and instruction-following. |
| 5 | What is machine learning and how is it related to artificial intelligence? | Machine learning is a field where systems learn patterns from data to improve task performance; it is a core subfield and practical approach within AI. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | AI capabilities | Đoạn nói về explanation của agent decisions (không phải định nghĩa AI trực tiếp) | 0.483 | Partial | Chưa chạy agent answer trong benchmark này |
| 2 | Meaning of "deep" | Đoạn lệch chủ đề, nói về artistic sensitivity/cognitive hierarchy | 0.320 | No | Chưa chạy agent answer trong benchmark này |
| 3 | Embedding + cosine similarity | Đoạn mô tả similarity measure và cosine cho embeddings | 0.248 | Yes | Chưa chạy agent answer trong benchmark này |
| 4 | LLMs are trained to do what | Đoạn thiên về discussion openness, chưa trả lời trực diện training objective | 0.332 | Partial | Chưa chạy agent answer trong benchmark này |
| 5 | ML and relation to AI | Đoạn nói về feature learning trong ML, khá gần gold answer | 0.390 | Yes | Chưa chạy agent answer trong benchmark này |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 3 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Mình học được cách thiết kế strategy theo intent truy vấn thay vì chỉ dựa vào separator cố định. Cách tiếp cận agentic chunking của bạn cùng nhóm cho thấy có thể cải thiện độ bám ngữ nghĩa ở các tài liệu dài nhiều mục con. Đây là hướng đáng thử khi baseline bắt đầu chạm trần.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Mình học được cách dùng metadata schema rõ ràng (`doc_id`, `source`, `language`) để debug retrieval nhanh hơn. Khi so sánh cùng query giữa filtered/unfiltered, nhóm bạn khác ghi log nguồn rất chi tiết nên dễ tìm nguyên nhân lỗi. Cách trình bày này giúp đánh giá grounding trực quan hơn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu làm lại, mình sẽ bổ sung bước làm sạch dữ liệu mạnh hơn (lọc citation/reference noise) trước khi index và tách bộ train benchmark theo độ khó query. Mình cũng sẽ thêm một custom chunker theo heading + paragraph windows để giảm các case top-1 lệch ý. Cuối cùng, mình sẽ chạy thêm embedding backend thực (local) để so sánh công bằng với strategy agentic.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **82 / 100** |
