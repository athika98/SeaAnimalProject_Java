/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ch.zhaw.deeplearningjava.seaanimals;

/// Imports ///
import ai.djl.Model;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
/// Imports ///

/** Helper Klasse zum laden und speichern des Models. */
public final class Models {

    // Anzahl der Klassifikationsetiketten, z.B. Aal, Fisch usw..
    public static final int NUM_OF_OUTPUT = 21;

    // Definieren der Höhe und Breite der Bilder für die Vorverarbeitung
    public static final int IMAGE_HEIGHT = 100;
    public static final int IMAGE_WIDTH = 100;

    // Name des Modells
    public static final String MODEL_NAME = "seaanimalclassifier";

    private Models() {}

    public static Model getModel() {

        // Erstellen einer neuen Modellinstanz mit dem angegebenen Namen
        Model model = Model.newInstance(MODEL_NAME);

        /// Erstellen des neuronalen Netzwerks mithilfe der Builder-Methode von ResNetV1
        Block resNet50 =
                ResNetV1.builder() // Starten des Aufbaus eines ResNet
                        .setImageShape(new Shape(3, IMAGE_HEIGHT, IMAGE_WIDTH)) // Festlegen der Bildgröße und -kanäle
                        .setNumLayers(50) // Anzahl der Schichten im Netzwerk
                        .setOutSize(NUM_OF_OUTPUT) // Festlegen der Anzahl der Ausgabeklassen
                        .build();

        // Zuweisen des erstellten Blocks (Netzwerks) zum Modell
        model.setBlock(resNet50);
        return model;
    }

    // Erstellen der Synset-Datei
    public static void saveSynset(Path modelDir, List<String> synset) throws IOException {
        Path synsetFile = modelDir.resolve("synset.txt");
        try (Writer writer = Files.newBufferedWriter(synsetFile)) {
            writer.write(String.join("\n", synset));
        }
    }
}