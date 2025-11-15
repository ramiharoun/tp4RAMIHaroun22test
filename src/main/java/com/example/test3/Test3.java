package com.example.test3;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import com.example.test2.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test3 {

    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);
        ConsoleHandler consoleHandler = new ConsoleHandler();
        consoleHandler.setLevel(Level.FINE);
        logger.addHandler(consoleHandler);
    }

    public static void main(String[] args) {

        configureLogger();
        System.out.println("\n=== Test 3 – Routage de sources RAG ===\n");

        DocumentParser parser = new ApacheTikaDocumentParser();
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        List<TextSegment> iaSegments = loadAndSplit(Paths.get("src/main/resources/support_rag.pdf"), parser);
        List<TextSegment> carSegments = loadAndSplit(Paths.get("src/main/resources/Controle_blanc.pdf"), parser);

        EmbeddingStore<TextSegment> iaStore = new InMemoryEmbeddingStore<>();
        EmbeddingStore<TextSegment> carStore = new InMemoryEmbeddingStore<>();

        iaStore.addAll(embeddingModel.embedAll(iaSegments).content(), iaSegments);
        carStore.addAll(embeddingModel.embedAll(carSegments).content(), carSegments);

        ContentRetriever iaRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(iaStore)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        ContentRetriever mathsRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(carStore)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("GEMINI_KEY doit être définie.");
        }

        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        Map<ContentRetriever, String> descriptions = new LinkedHashMap<>();
        descriptions.put(iaRetriever, "Supports de cours décrivant le RAG, les LLM et des notions d’IA.");
        descriptions.put(mathsRetriever, "C'est un controle blanc de mathématique de 1ere année");

        var router = new LanguageModelQueryRouter(chatModel, descriptions);

        var augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("Pose une question (tape 'exit' pour quitter) :");
            while (true) {
                System.out.print("> ");
                String question = scanner.nextLine();
                if ("exit".equalsIgnoreCase(question)) break;
                if (question.isBlank()) continue;

                String answer = assistant.chat(question);
                System.out.println("\n" + answer + "\n");
            }
        }
    }

    private static List<TextSegment> loadAndSplit(Path path, DocumentParser parser) {
        Document document = FileSystemDocumentLoader.loadDocument(path, parser);
        return DocumentSplitters.recursive(260, 40).split(document);
    }
}

