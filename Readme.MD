```mermaid
graph TD
    %% Define Nodes and Shapes
    start([User Uploads Image])
    graph TD
    %% Define Nodes and Shapes
    start([User Uploads Image])
    upscale{Step 1: Bicubic Upscale};
    parallel_split((Split for Dual Analysis));
    
    subgraph Supervised[Supervised Classification (MobileNetV2)]
        mobilenet[Step 2: MobileNetV2 Prediction];
        c_model{Output: C_model & Species_model};
    end
    
    subgraph Unsupervised[Unsupervised Verification (Feature Space)]
        embedding[Step 3: Feature Embedding Extraction];
        cosine[Cosine Similarity vs. Stored Embeddings (.pkl)];
        c_sim{Output: C_sim & Species_sim};
    end
    
    fusion[Step 4: Confidence Fusion];
    formula(C_final = 0.5 * C_model + 0.5 * C_sim);
    decision{Step 5: Decision Logic};
    
    condition_check{C_final $\ge$ 0.6 AND C_sim $\ge$ 0.8};
    
    accept[✅ ACCEPTED: Final Species Name];
    
    gemini[Step 6: Gemini API / DB Retrieval];
    
    report((Final Report Display));
    
    reject[❌ REJECTED: Unknown Species / New Candidate];
    
    end_process([END]);
    
    %% Define Flow
    start --> upscale;
    upscale --> parallel_split;
    
    %% Dual Path
    parallel_split --> mobilenet;
    mobilenet --> c_model;
    
    parallel_split --> embedding;
    embedding --> cosine;
    cosine --> c_sim;
    
    %% Rejoin and Fusion
    c_model & c_sim --> fusion;
    fusion --> formula;
    formula --> decision;
    
    %% Decision Logic
    decision --> condition_check;
    
    condition_check -- YES --> accept;
    condition_check -- NO --> reject;
    
    %% Final Retrieval
    accept --> gemini;
    gemini --> report;
    
    reject --> end_process;
    report --> end_process;

    %% Style Formatting for Clarity
    classDef start_end fill:#E0F7FA, stroke:#00BCD4, stroke-width:2px;
    class start, end_process start_end;
    classDef process fill:#C8E6C9, stroke:#4CAF50, stroke-width:2px;
    class upscale, mobilenet, embedding, cosine, fusion, gemini process;
    classDef output fill:#FFECB3, stroke:#FFC107, stroke-width:2px;
    class c_model, c_sim output;
    classDef decision_node fill:#FFCDD2, stroke:#F44336, stroke-width:2px;
    class decision, condition_check decision_node;
    classDef final_accept fill:#A5D6A7, stroke:#388E3C, stroke-width:2px;
    class accept final_accept;
    classDef final_reject fill:#EF9A9A, stroke:#D32F2F, stroke-width:2px;
    class reject final_reject;
    classDef report_node fill:#BBDEFB, stroke:#2196F3, stroke-width:2px;
    class report report_node;
```
