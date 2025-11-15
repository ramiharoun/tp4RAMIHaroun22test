package com.example.test4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
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
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class Test4 {

    public static void main(String[] args) {

        DocumentParser parser = new ApacheTikaDocumentParser();
        Path pdf = Paths.get("src/main/resources/Controle_blanc.pdf");
        Document doc = FileSystemDocumentLoader.loadDocument(pdf, parser);

        List<TextSegment> slices = DocumentSplitters.recursive(280, 40).split(doc);

        EmbeddingModel embedder = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> vectors = embedder.embedAll(slices).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(vectors, slices);

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        var retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embedder)
                .maxResults(2)
                .minScore(0.5)
                .build();

        AssistantLimite assistant = AiServices.builder(AssistantLimite.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(retriever)
                .build();

        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print("Pose une question (tape 'exit' pour quitter) : ");
            String q = sc.nextLine();
            if (q.equalsIgnoreCase("exit")) break;
            System.out.println(assistant.chat(q) + "\n");
        }
    }
}
