package ch.zhaw.deeplearningjava.seaanimals;

/// Imports ///
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
/// Imports ///

/**
 * In training, multiple passes (or epochs) are made over the training data
 * trying to find patterns
 * and trends in the data, which are then stored in the model. During the
 * process, the model is
 * evaluated for accuracy using the validation data. The model is updated with
 * findings over each
 * epoch, which improves the accuracy of the model.
 */
public final class Training {

    // Anzahl der Trainingsbeispiele, die verarbeitet werden, bevor das Modell aktualisiert wird
    private static final int BATCH_SIZE = 32;

    // Anzahl der Durchläufe über den kompletten Datensatz
    private static final int EPOCHS = 11;

    public static void main(String[] args) throws IOException, TranslateException {
        // Pfad, wo das Modell gespeichert wird
        Path modelDir = Paths.get("models");

        // Erstellung des Datensatzes aus einem Verzeichnis
        ImageFolder dataset = initDataset("images/root");
        // Aufteilung des Datensatzes in Trainings- und Validierungsdaten
        RandomAccessDataset[] datasets = dataset.randomSplit(8, 2);

        // Festlegen der Verlustfunktion, die während des Trainings minimiert werden soll
        Loss loss = Loss.softmaxCrossEntropyLoss();

        // Konfiguration der Trainingseinstellungen
        TrainingConfig config = setupTrainingConfig(loss);

        // Erstellung des Modells und Initialisierung des Trainers
        Model model = Models.getModel(); 
        Trainer trainer = model.newTrainer(config);
        trainer.setMetrics(new Metrics());

        // Festlegen der Eingabeform für den Trainer
        Shape inputShape = new Shape(1, 3, Models.IMAGE_HEIGHT, Models.IMAGE_HEIGHT);
        trainer.initialize(inputShape);

        // Training des Modells
        EasyTrain.fit(trainer, EPOCHS, datasets[0], datasets[1]);

        // Ergebnisse des Trainings extrahieren und im Modell speichern
        TrainingResult result = trainer.getTrainingResult();
        model.setProperty("Epoch", String.valueOf(EPOCHS));
        model.setProperty(
                "Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
        model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));

        // Speichern des trainierten Modells und der Klassennamen (Labels)
        model.save(modelDir, Models.MODEL_NAME);
        Models.saveSynset(modelDir, dataset.getSynset());
    }

    private static ImageFolder initDataset(String datasetRoot)
            throws IOException, TranslateException {
        // Erstellung eines Bildordners als Datensatz
        ImageFolder dataset = ImageFolder.builder()
                // retrieve the data
                .setRepositoryPath(Paths.get(datasetRoot))
                .optMaxDepth(10)
                .addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))
                .addTransform(new ToTensor())
                // random sampling; don't process the data in order
                .setSampling(BATCH_SIZE, true)
                .build();

        dataset.prepare();
        return dataset;
    }

    private static TrainingConfig setupTrainingConfig(Loss loss) {
        return new DefaultTrainingConfig(loss)
                // using Adam optimizer with default settings
                .optOptimizer(Optimizer.adam().build())
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }
}
