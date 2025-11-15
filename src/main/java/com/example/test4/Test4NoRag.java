package com.example.test4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class Test4NoRag {

    public static void main(String[] args) {

        System.out.println("------------Test 4 : Activation conditionnelle du RAG ------------");

        Document base = FileSystemDocumentLoader.loadDocument(
                Paths.get("src/main/resources/Controle_blanc.pdf"),
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> parts = DocumentSplitters.recursive(280, 35).split(base);

        EmbeddingModel embed = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> vectors = embed.embedAll(parts).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(vectors, parts);

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();

        var retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embed)
                .maxResults(2)
                .minScore(0.5)
                .build();

        QueryRouter router = query -> {
            String task = """
                    Cette demande parle-t-elle de RAG, embeddings, LLM, fine-tuning ou IA ?
                    Réponds uniquement oui ou non.

                    %s
                    """.formatted(query.text());

            String res = model.chat(task).toLowerCase();
            return res.contains("oui") ? List.of(retriever) : List.of();
        };

        RetrievalAugmentor aug = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        var assistant = AiServices.builder(AssistantLimite.class)
                .chatModel(model)
                .retrievalAugmentor(aug)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print("Pose une question (tape 'exit' pour quitter) : ");
            String q = sc.nextLine();
            if (q.equalsIgnoreCase("exit")) break;
            System.out.println(assistant.chat(q));
        }
    }
}
