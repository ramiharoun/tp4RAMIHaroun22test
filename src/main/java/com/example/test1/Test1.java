package com.example.test1;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;

import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.chat.ChatModel;

import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStore;

import java.nio.file.*;
import java.util.*;

public class Test1 {

    public static void main(String[] args) {

        System.out.println("----------- RAG Test 1 -----------");

        Path pdfPath = Paths.get("src/main/resources/support_rag.pdf");
        Document pdf = FileSystemDocumentLoader.loadDocument(pdfPath, new ApacheTikaDocumentParser());

        var splitter = DocumentSplitters.recursive(140, 25);
        List<TextSegment> chunks = splitter.split(pdf);

        EmbeddingModel embedder = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> chunkVectors = embedder.embedAll(chunks).content();

        EmbeddingStore<TextSegment> index = new InMemoryEmbeddingStore<>();
        index.addAll(chunkVectors, chunks);

        String api = System.getenv("GEMINI_KEY");
        if (api == null || api.isEmpty()) {
            throw new RuntimeException("Variable d'environnement GEMINI_KEY absente.");
        }

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(api)
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .build();

        var retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingModel(embedder)
                .embeddingStore(index)
                .maxResults(3)
                .minScore(0.3)
                .build();

        var history = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .contentRetriever(retriever)
                .chatMemory(history)
                .build();

        Scanner input = new Scanner(System.in);
        System.out.println("Pose une question (tape 'exit' pour terminer)");

        while (true) {
            System.out.print("> ");
            String question = input.nextLine().trim();
            if (question.equalsIgnoreCase("exit")) break;
            if (question.isEmpty()) continue;

            String reply = assistant.chat(question);
            System.out.println("\n" + reply + "\n");
        }
    }
}
