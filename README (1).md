# RAG Sıfırdan Uygulamalı — Chunking Stratejileri Karşılaştırması

LangChain v1 ile uçtan uca bir **RAG (Retrieval-Augmented Generation)** sistemi: veriyi yükleme, chunking, embedding, retrieval, cevap üretimi ve değerlendirme. Projenin odağında **chunking stratejilerinin aynı veri üzerinde ölçülerek karşılaştırılması** var.

📓 **Kaggle (çıktılarıyla):** [Kaggle linki]

## Senaryo

Hayali bir akvaryum şirketi olan *Mavi Resif Akvaryum* için bir **müşteri destek asistanı**. Asistan dört dokümana dayanarak cevap veriyor:

| Doküman | Yapı | Neden seçildi |
|---|---|---|
| `bakim_rehberi.md` | Başlıklı, tablolu | Layout-aware chunking'i test etmek |
| `hastaliklar.md` | Başlıklı, kısa bölümler | Bölüm bazlı bölmeyi test etmek |
| `iade_politikasi.txt` | Düz metin | Yapısız metinde recursive bölmeyi test etmek |
| `musteri_gorusmesi.txt` | Konudan konuya atlayan transkript | Semantic chunking'i test etmek |

Asistan cevaplarında kaynak gösteriyor; bilgi bankasında olmayan bir soru gelince bilmediğini söylüyor.

> Tüm veriler hayalidir ve notebook tarafından üretilir.

## Notebook'un içeriği

| Adım | Konu |
|---|---|
| 1-2 | Örnek bilgi bankası ve `Document` yapısı |
| 3 | Chunking laboratuvarı: Naive, **Recursive**, **Layout-Aware** (Markdown), **Semantic** (elle + `SemanticChunker`), **Agentic** (LLM ile) |
| 3g | Atölye: token bazlı bölme (`from_huggingface_tokenizer`, `from_tiktoken_encoder`), kod (`from_language`), HTML (`HTMLHeaderTextSplitter`, `HTMLSemanticPreservingSplitter`), JSON (`RecursiveJsonSplitter`) |
| 4 | Embedding + vektör veritabanı, **Small-to-Big** (`ParentDocumentRetriever`), `MultiVectorRetriever` (varsayımsal sorular) |
| 5 | Retriever ayarları: similarity, MMR, metadata filtresi |
| 6 | Kaynak gösteren cevap zinciri (LCEL) |
| 7 | Chunking stratejilerinin **Hit@3** ile karşılaştırılması |
| 8 | Hybrid search (BM25 + vektör) |
| 9 | Aynı asistanın **CAG** (Cache-Augmented Generation) ile kurulması ve RAG ile karşılaştırma |

## Chunking yöntemleri özeti

| Yöntem | Temel fikir | En uygun olduğu yer |
|---|---|---|
| Recursive | Metin yapısına göre hiyerarşik bölme | Hızlı prototip, genel metinler |
| Semantic | Anlam akışındaki kırılmalara göre bölme | Transkript, blog, anlatısal metin |
| Layout-Aware | Dokümanın başlık/format yapısına göre bölme | Kurumsal raporlar, kılavuzlar, PDF, HTML |
| Small-to-Big | Küçük parçalarla ara, büyük parçayı LLM'e ver | Üretimdeki çoğu RAG sistemi |
| Agentic | LLM'e böldürme | Az sayıda, çok değerli doküman |

## Çıkarımlar

- **RAG, fine-tuning'in ileri seviyesi değil; alternatifi.** Fine-tuning modelin nasıl konuşacağını, RAG neyi bileceğini belirler.
- **Kötü cevapların çoğu retrieval'dan kaynaklanır.** Hata ayıklamaya önce getirilen chunk'lara bakarak başlanmalı.
- **Naif bölme tabloları satır ortasından kesebilir.** Başlık satırından kopan değerler anlamını yitirir.
- **Chunk boyutu embedding modelinin tokenizer'ıyla ölçülmeli.** Limiti aşan kısım sessizce kırpılır.
- **Türkçe JSON verisinde `ensure_ascii=False` şart.** Aksi halde karakterler kaçış koduna dönüşür.
- **En iyi chunking stratejisi, kendi verinde ölçülerek seçilir.**

## Çalıştırma

### Kaggle
1. Notebook'u Kaggle'da açıp **Copy & Edit** de.
2. *Settings* → **Internet: On**.
3. *Add-ons* → **Secrets** → `ANTHROPIC_API_KEY` ekle ve notebook'a bağla.
4. **Run All**.

### Yerel
```bash
git clone <bu-repo>
cd <bu-repo>
pip install -r requirements.txt
cp .env.example .env        # .env içine kendi ANTHROPIC_API_KEY değerini yaz
jupyter notebook rag_uygulamali.ipynb
```

> Chunking, embedding ve retrieval hücreleri API anahtarı olmadan da çalışır (embedding modeli yereldir). Anahtar yalnızca LLM kullanan hücreler için gerekir: agentic chunking, MultiVector, cevap üretimi ve CAG.

## Kullanılan teknolojiler

LangChain v1 (`langchain-core`, `langchain-text-splitters`, `langchain-classic`) · Claude (Anthropic) · `intfloat/multilingual-e5-base` embedding modeli · BM25 · Kaggle

## Sırada

LangGraph ile **agentic RAG**: soru bilgi bankası gerektiriyor mu, gelen chunk'lar yeterli mi, değilse soruyu yeniden yazıp tekrar ara.
