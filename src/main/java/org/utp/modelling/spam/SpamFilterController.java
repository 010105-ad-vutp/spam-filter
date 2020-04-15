package org.utp.modelling.spam;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;

@RestController
public class SpamFilterController {

  private final SpamFilterService spamFilterService;

  @Autowired
  public SpamFilterController(SpamFilterService spamFilterService) {
    this.spamFilterService = spamFilterService;
  }

  @PostMapping("/spam-filter")
  public Double spamFilter(@RequestBody String message) {
    String decodedMessage = decode(message);
    return spamFilterService.testMessageForSpam(decodedMessage);
  }

  private String decode(String message) {
    try {
      return URLDecoder.decode(message, "UTF-8");
    } catch(UnsupportedEncodingException e) {
      throw new RuntimeException(e);
    }
  }
}
