class TweetBlock {
  double _polarity = 0.0;
  double _subjectivity = 0.0;
  String _date = '';
  int _index = 0;
  String _image = '';
  List<String> _entities = [];
  List<String> _tweets = [];

  TweetBlock(double polarity, double subjectivity, String date, int index,
      String image, List<String> entities, List<String> tweets) {
    _polarity = polarity;
    _subjectivity = subjectivity;
    _date = date;
    _index = index;
    _image = image;
    _entities = entities;
    _tweets = tweets;
  }

  double getPolarity() {
    return _polarity;
  }

  double getSubjectivity() {
    return _subjectivity;
  }

  String getDate() {
    return _date;
  }

  int getIndex() {
    return _index;
  }

  String getImageURL() {
    return _image;
  }

  List<String> getEntities() {
    return _entities;
  }

  List<String> getTweets() {
    return _tweets;
  }
}
