using System;
using System.Collections.Generic;
using System.Linq;
using OpenUtau.Api;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.Neutrino {
    /// <summary>
    /// Minimal phonemizer for NEUTRINO: passes lyrics through as-is.
    /// The actual phoneme conversion happens in NeutrinoRenderer.
    /// </summary>
    [Phonemizer("NEUTRINO Phonemizer", "NEUTRINO", language: "JA")]
    public class NeutrinoPhonemizer : Phonemizer {
        public override void SetSinger(USinger singer) {
            // No-op: NEUTRINO renderer handles phoneme conversion itself
        }

        public override Result Process(Note[] notes, Note? prev, Note? next,
            Note? prevNeighbour, Note? nextNeighbour, Note[] prevs) {

            var lyric = notes[0].lyric ?? "R";

            // Just pass the lyric through as a single phoneme.
            // NeutrinoRenderer reads phrase.notes[i].lyric directly
            // and converts via NeutrinoPhoneme.KanaToPhonemeIds().
            return new Result {
                phonemes = new Phoneme[] {
                    new Phoneme {
                        phoneme = lyric,
                        position = 0,
                    }
                }
            };
        }
    }
}
