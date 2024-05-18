package ch.zhaw.deeplearningjava.seaanimals;

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

@RestController
public class ClassificationController {

    private Inference inference = new Inference();

    @GetMapping("/ping")
    public String ping() {
        return "Classification app is up and running!";
    }

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

        // Sort the results by probability in descending order
        Collections.sort(results, new Comparator<Map<String, Object>>() {
            @Override
            public int compare(Map<String, Object> o1, Map<String, Object> o2) {
                return Double.compare((double) o2.get("probability"), (double) o1.get("probability"));
            }
        });

        // Convert the image to a Base64 string
        String base64Image = java.util.Base64.getEncoder().encodeToString(image.getBytes());

        // Create the response JSON with the image URL and classification results
        Map<String, Object> response = new HashMap<>();
        response.put("image", base64Image);
        response.put("results", results);

        return ResponseEntity.ok().body(new Gson().toJson(response));
    }
}
