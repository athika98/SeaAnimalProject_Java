package ch.zhaw.deeplearningjava.seaanimals;

/// Imports ///
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.google.gson.Gson;

import ai.djl.modality.Classifications;
/// Imports ///

@RestController
public class ClassificationController {

    private Inference inference = new Inference();

    @GetMapping("/ping")
    public String ping() {
        return "Classification app is up and running!";
    }

    // POST-Methode zur Analyse von Bildern und Klassifizierung von Meerestieren
    @PostMapping(path = "/analyze")
    public ResponseEntity<String> predict(@RequestParam("image") MultipartFile image) throws Exception {
        System.out.println(image.getOriginalFilename());
        Classifications classifications = inference.predict(image.getBytes());
        List<Map<String, Object>> results = new ArrayList<>();
        for (Classifications.Classification classification : classifications.items()) {
            Map<String, Object> result = new HashMap<>();
            result.put("className", classification.getClassName().replace(" ", "_"));
            result.put("probability", classification.getProbability());
            results.add(result);
        }

        // Sortierung der Ergebnisse nach Wahrscheinlichkeit in absteigender Reihenfolge
        Collections.sort(results, new Comparator<Map<String, Object>>() {
            @Override
            public int compare(Map<String, Object> o1, Map<String, Object> o2) {
                return Double.compare((double) o2.get("probability"), (double) o1.get("probability"));
            }
        });

        // Konvertierung des Bildes in einen Base64-kodierten String für die einfache Übertragung
        String base64Image = java.util.Base64.getEncoder().encodeToString(image.getBytes());

        // Erstellung der Antwort im JSON-Format mit dem Bild und den Klassifikationsergebnissen
        Map<String, Object> response = new HashMap<>();
        response.put("image", base64Image);
        response.put("results", results);

        // Rückgabe der Antwort als JSON-String im Body der HTTP-Antwort
        return ResponseEntity.ok().body(new Gson().toJson(response));
    }
}
