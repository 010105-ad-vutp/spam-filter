package org.utp.modelling.spam;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;

public class EmailDatasetIterator implements DataSetIterator {

  private final WordVectors wordVectors;
  private final int batchSize;
  private final int vectorSize;
  private final int truncateLength;

  private final List<File> spamFiles;
  private final List<File> nonSpamFiles;
  private final TokenizerFactory tokenizerFactory;

  private int spamCursor = 0;
  private int nonSpamCursor = 0;
  private double spamNonSpamRatio;

  public EmailDatasetIterator(String baseDir, WordVectors vectors, int batchSize, int truncateLength) {
    this.wordVectors = vectors;
    this.batchSize = batchSize;
    this.vectorSize = vectors.getWordVector(vectors.vocab().wordAtIndex(0)).length;
    this.truncateLength = truncateLength;

    this.spamFiles = Arrays.asList(new File(this.getClass().getResource(baseDir + "spam/").getPath()).listFiles());
    this.nonSpamFiles = Arrays.asList(new File(this.getClass().getResource(baseDir + "non-spam/").getPath()).listFiles());
    this.tokenizerFactory = new DefaultTokenizerFactory();
    this.tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

    this.spamNonSpamRatio = spamFiles.size() / (double) nonSpamFiles.size();
  }

  public INDArray loadFeaturesFromString(String email, int maxLength) {
    List<String> tokens = tokenize(email);

    int outputLength = Math.min(maxLength, tokens.size());
    INDArray features = Nd4j.create(1, vectorSize, outputLength);

    int index = 0;
    while (index++ < outputLength - 1) {
      extractFeatures(tokens.get(index), features, index);
    }

    return features;
  }

  private void extractFeatures(String token, INDArray features, int index) {
    INDArray vector = wordVectors.getWordVectorMatrix(token);
    if (vector == null) return;

    features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(index)}, vector);
  }

  private List<String> tokenize(String text) {
    return tokenizerFactory.create(text).getTokens()
        .stream()
        .filter(wordVectors::hasWord)
        .collect(Collectors.toList());
  }

  @Override
  public DataSet next(int i) {
    if (spamCursor == spamFiles.size() && nonSpamCursor == nonSpamFiles.size()) {
      throw new NoSuchElementException("Data set has no more elements");
    }

    try {
      return getNext(i);
    } catch (IOException e) {
      throw new RuntimeException("Failed retrieving next element", e);
    }
  }

  private DataSet getNext(int size) throws IOException {
    List<String> emails = new ArrayList<>(size);
    boolean[] spamStatus = new boolean[size];

    int spamStatusIndex = 0;

    if (spamCursor < spamFiles.size()) {
      spamStatusIndex = (int) Math.floor(spamNonSpamRatio * batchSize) - 1;
      int index = 0;
      while (index < spamStatusIndex && spamCursor < spamFiles.size()) {
        String text = FileUtils.readFileToString(spamFiles.get(spamCursor++), StandardCharsets.UTF_8);
        emails.add(text);
        spamStatus[index++] = true;
      }
    }

    int index = spamStatusIndex;
    while (index < batchSize && nonSpamCursor < nonSpamFiles.size()) {
      String text = FileUtils.readFileToString(nonSpamFiles.get(nonSpamCursor++), StandardCharsets.UTF_8);
      emails.add(text);
      spamStatus[index++] = false;
    }

    List<List<String>> tokens = new ArrayList<>(emails.size());
    int maxLength = 0;
    for (String s : emails) {
      List<String> currentTokens = tokenize(s);

      tokens.add(currentTokens);
      maxLength = Math.max(maxLength, currentTokens.size());
    }

    if (maxLength > truncateLength) maxLength = truncateLength;

    INDArray features = Nd4j.create(emails.size(), vectorSize, maxLength);
    INDArray labels = Nd4j.create(emails.size(), 2, maxLength);
    INDArray featuresMask = Nd4j.zeros(emails.size(), maxLength);
    INDArray labelsMask = Nd4j.zeros(emails.size(), maxLength);

    for (int i = 0; i < emails.size() - 1; i++) {
      List<String> currentTokens = tokens.get(i);
      int length = Math.min(currentTokens.size(), maxLength);

      for (int j = 0; j < length; j++) {
        String token = currentTokens.get(j);
        INDArray vector = wordVectors.getWordVectorMatrix(token);
        features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
        featuresMask.get(NDArrayIndex.point(i), NDArrayIndex.interval(0, length)).assign(1);
      }

      int idx = spamStatus[i] ? 1 : 0;
      int lastIdx = Math.min(currentTokens.size(), maxLength);
      labels.putScalar(new int[]{i, idx, lastIdx - 1}, 1.0);
      labelsMask.putScalar(new int[]{i, lastIdx - 1}, 1.0);
    }

    return new DataSet(features, labels, featuresMask, labelsMask);
  }

  @Override
  public int inputColumns() {
    return 0;
  }

  @Override
  public int totalOutcomes() {
    return 2;
  }

  @Override
  public boolean resetSupported() {
    return true;
  }

  @Override
  public boolean asyncSupported() {
    return true;
  }

  @Override
  public void reset() {
    spamCursor = 0;
    nonSpamCursor = 0;
  }

  @Override
  public int batch() {
    return batchSize;
  }

  @Override
  public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
    throw new UnsupportedOperationException();
  }

  @Override
  public DataSetPreProcessor getPreProcessor() {
    throw new UnsupportedOperationException();
  }

  @Override
  public List<String> getLabels() {
    return Arrays.asList("non-spam", "spam");
  }

  @Override
  public boolean hasNext() {
    return spamCursor + nonSpamCursor < spamFiles.size() + nonSpamFiles.size();
  }

  @Override
  public DataSet next() {
    return next(batchSize);
  }
}
