package com.example.test1;

import dev.langchain4j.service.SystemMessage;

public interface Assistant {

    @SystemMessage("Ton rôle est d'agir comme un assistant basé sur RAG : tes réponses doivent provenir exclusivement des informations contenues dans le document PDF.")
    String chat(String userMessage);
}