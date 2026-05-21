"""
Functionality test for TextToAudioStream sentence grouping.
"""

from RealtimeTTS import TextToAudioStream


def assert_grouped(sentences, group_size, expected):
    grouped = list(TextToAudioStream._group_sentences(iter(sentences), group_size))
    assert grouped == expected, f"expected {expected!r}, got {grouped!r}"


def main():
    assert_grouped(
        ["One.", "Two.", "Three."],
        1,
        ["One.", "Two.", "Three."],
    )
    assert_grouped(
        ["One.", "Two.", "Three.", "Four.", "Five."],
        2,
        ["One. Two.", "Three. Four.", "Five."],
    )
    assert_grouped(
        [" One. ", "", "  ", "Two.\n", "\tThree."],
        2,
        ["One. Two.", "Three."],
    )
    assert_grouped(
        ["One.", "Two.", "Three.", "Four.", "Five.", "Six."],
        3,
        ["One. Two. Three.", "Four. Five. Six."],
    )

    print("group_sentences functionality test passed")


if __name__ == "__main__":
    main()
