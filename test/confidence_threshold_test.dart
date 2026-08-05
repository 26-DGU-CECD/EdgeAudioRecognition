import 'package:flutter_test/flutter_test.dart';
import 'package:untitled/models/sound_packet.dart';

SoundPacket _packet({
  required String label,
  String? displayLabel,
  required double score,
}) {
  return SoundPacket.fromJson({
    'status': 'ok',
    'time': '12:00:00',
    'label': label,
    'display_label': ?displayLabel,
    'score': score,
    'db': 60.0,
    'level': 'caution',
    'angle': 90.0,
    'has_doa': true,
  });
}

const _allRawLabels = [
  'car_horn',
  'siren',
  'cat',
  'bicycle_bell',
  'water',
  'dog_bark',
  'gunshot',
  'scream',
  'glass_break',
  'fire_alarm',
  'baby_cry',
  'knock',
  'cry',
];

void main() {
  test('임계값은 클래스 구분 없이 85%로 균일하다', () {
    expect(kMinConfidenceScore, 0.85);
    for (final rawLabel in _allRawLabels) {
      expect(minConfidenceScoreFor(rawLabel), 0.85, reason: rawLabel);
    }
  });

  group('85% 이상만 통과한다', () {
    for (final rawLabel in _allRawLabels) {
      test(rawLabel, () {
        expect(_packet(label: rawLabel, score: 0.86).isConfident, isTrue);
        expect(_packet(label: rawLabel, score: 0.85).isConfident, isTrue);
        expect(_packet(label: rawLabel, score: 0.84).isConfident, isFalse);
      });
    }
  });

  test('bicycle_bell은 자전거로 표시된다', () {
    expect(displayNameForSound('bicycle_bell'), '자전거');
    expect(displayNameForSound('bike_bell'), '자전거');
  });

  test('status가 ok가 아니면 점수와 무관하게 걸러진다', () {
    final packet = SoundPacket.fromJson({
      'status': 'error',
      'label': 'car_horn',
      'score': 0.99,
      'db': 60.0,
    });
    expect(packet.isConfident, isFalse);
  });
}
