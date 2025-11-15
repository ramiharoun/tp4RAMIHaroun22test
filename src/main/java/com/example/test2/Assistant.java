package com.example.test2;


    import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
    import dev.langchain4j.service.V;

public interface Assistant {

        @SystemMessage("""
        Ton rôle est d’accompagner l’utilisateur avec des explications précises et accessibles.
        Présente les idées de manière ordonnée et facile à suivre.
        Ajoute, si nécessaire, de courts exemples illustratifs.
        Toutes les réponses doivent être en français.
        """)
        @UserMessage("Demande : {{message}}")
        String chat(@V("message") String message);
    }

