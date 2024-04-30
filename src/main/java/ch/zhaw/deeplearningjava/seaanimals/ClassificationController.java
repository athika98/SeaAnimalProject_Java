package ch.zhaw.deeplearningjava.seaanimals;

import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Path;
import java.nio.file.Paths;

@RestController
public class ClassificationController {

    private Inference inference = new Inference();

    @GetMapping("/ping")
    public String ping() {
        return "Classification app is up and running!";
    }

    @PostMapping(path = "/analyze")
    public String predict(@RequestParam("image") MultipartFile image) throws Exception {
        System.out.println(image);
        // Annehmen, dass die predict-Methode auch den Klassennamen zurückgibt
        return inference.predict(image.getBytes()).toJson();
    }

    // Endpunkt zum Abrufen der Bilder basierend auf dem Klassennamen
    @GetMapping("/images/{className}")
    public ResponseEntity<Resource> getImage(@PathVariable String className) {
        Path imagePath = Paths.get("src/main/resources/static/images", className + ".png");
        try {
            Resource image = new UrlResource(imagePath.toUri());
            if (!image.exists() || !image.isReadable()) {
                throw new RuntimeException("Unable to read the image file");
            }
            return ResponseEntity
                    .ok()
                    .contentType(MediaType.IMAGE_JPEG)
                    .body(image);
        } catch (Exception e) {
            return ResponseEntity
                    .notFound()
                    .build();
        }
    }
}
