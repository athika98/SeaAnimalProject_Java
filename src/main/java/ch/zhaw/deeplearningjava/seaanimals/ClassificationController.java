package ch.zhaw.deeplearningjava.seaanimals;

import java.util.ArrayList;
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

    /*
     * @PostMapping(path = "/analyze")
     * public String predict(@RequestParam("image") MultipartFile image) throws
     * Exception {
     * System.out.println(image);
     * return inference.predict(image.getBytes()).toJson();
     * }
     */

    @PostMapping(path = "/analyze")
    public ResponseEntity<String> predict(@RequestParam("image") MultipartFile image) throws Exception {
        System.out.println(image.getOriginalFilename());
        Classifications classifications = inference.predict(image.getBytes());
        List<Map<String, Object>> results = new ArrayList<>();
        for (Classifications.Classification classification : classifications.items()) {
            Map<String, Object> result = new HashMap<>();
            result.put("className", classification.getClassName());
            result.put("probability", classification.getProbability());
            results.add(result);
        }
        return ResponseEntity.ok().body(new Gson().toJson(results));
    }

}