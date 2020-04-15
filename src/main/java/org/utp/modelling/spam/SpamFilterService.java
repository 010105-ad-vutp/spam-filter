package org.utp.modelling.spam;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.File;

@Service
public class SpamFilterService {

  private static final String WORD_VECTOR_FILE = "GoogleNews-vectors-negative300.bin.gz";
  private static final int WORD_VECTORS_LENGTH = 300;

  private static final int SEED = 1234;
  private static final int EPOCHS = 1;
  private static final int CLASSES = 2;
  private static final int BATCH_SIZE = 64;
  private static final int TRUNCATED_LENGTH = 120;

  private EmailDatasetIterator testIterator;
  private MultiLayerNetwork network;

  @PostConstruct
  private void init() {
    File vectorsFile = new File(this.getClass().getResource("/" + WORD_VECTOR_FILE).getPath());
    WordVectors wordVectors = WordVectorSerializer.loadStaticModel(vectorsFile);
    EmailDatasetIterator trainIterator = new EmailDatasetIterator(
        "/data/train/",
        wordVectors,
        BATCH_SIZE,
        TRUNCATED_LENGTH
    );

    this.network = createNetwork();
    network.init();
    network.setListeners(new StatsListener(new InMemoryStatsStorage()));
    network.fit(trainIterator, EPOCHS);

    testIterator = new EmailDatasetIterator(
        "/data/test/",
        wordVectors,
        1,
        TRUNCATED_LENGTH
    );
  }

  public Double testMessageForSpam(String message) {
    INDArray features = testIterator.loadFeaturesFromString(message, TRUNCATED_LENGTH);
    INDArray networkOutput = network.output(features);

    long size = networkOutput.size(2);
    INDArray probabilitiesAtLastWord = networkOutput.get(
        NDArrayIndex.point(0),
        NDArrayIndex.all(),
        NDArrayIndex.point(size - 1)
    );

    return probabilitiesAtLastWord.getDouble(1);
  }

  private MultiLayerNetwork createNetwork() {
    MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
        .seed(SEED)
        .updater(new Adam(5e-3))
        .l2(1e-5)
        .weightInit(WeightInit.XAVIER)
        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
        .list()
        .layer(new LSTM.Builder()
            .nIn(WORD_VECTORS_LENGTH)
            .nOut(TRUNCATED_LENGTH)
            .activation(Activation.TANH)
            .build())
        .layer(new RnnOutputLayer.Builder()
            .nIn(TRUNCATED_LENGTH)
            .nOut(CLASSES)
            .activation(Activation.SOFTMAX)
            .lossFunction(LossFunctions.LossFunction.MCXENT)
            .build())
        .build();

    return new MultiLayerNetwork(config);
  }
}
