using System.IO;
using System.Text;
using OpenUtau.Core.Ustx;
using Xunit;
using Xunit.Abstractions;

namespace OpenUtau.Classic {
    public class VoicebankLoaderTest {
        readonly ITestOutputHelper output;

        public VoicebankLoaderTest(ITestOutputHelper output) {
            this.output = output;
        }

        [Fact]
        public void OtoSetRoundTrip() {
            string text = @"a.wav=,,,,,
a.wav=- a,
a.wav=a R,500,,,
!@#$!@#$
aoieu.wav=- a,,,,,,,,,

aoieu.wav=a o,,
aoieu.wav=o i,,,
aoieu.wav=i e,,100,150,,,
aoieu.wav=e u,20,
aoieu.wav=u R,5,,33,44,,
".Replace("\r\n", "\n");
            string expected = @"a.wav=a,,,,,
a.wav=- a,,,,,
a.wav=a R,500,,,,
!@#$!@#$
aoieu.wav=- a,,,,,

aoieu.wav=a o,,,,,
aoieu.wav=o i,,,,,
aoieu.wav=i e,,100,150,,
aoieu.wav=e u,20,,,,
aoieu.wav=u R,5,,33,44,
".Replace("\r\n", "\n");

            using (MemoryStream stream = new MemoryStream(Encoding.ASCII.GetBytes(text))) {
                VoicebankLoader.IsTest = true;
                var otoSet = VoicebankLoader.ParseOtoSet(stream, "oto.ini", Encoding.ASCII);
                using (MemoryStream stream2 = new MemoryStream()) {
                    VoicebankLoader.WriteOtoSet(otoSet, stream2, Encoding.ASCII);
                    string actual = Encoding.ASCII.GetString(stream2.ToArray());
                    Assert.Equal(expected, actual);
                }
            }
        }

        [Fact]
        public void DetectsNeutrinoModelDespiteGenericUstxConfig() {
            string dir = Path.Combine(Path.GetTempPath(), $"neutrino-detect-{System.Guid.NewGuid():N}");
            try {
                Directory.CreateDirectory(Path.Combine(dir, "model"));
                string characterPath = Path.Combine(dir, "character.txt");
                File.WriteAllText(characterPath, "name=Test NEUTRINO\n", Encoding.UTF8);
                using (var stream = File.Create(Path.Combine(dir, "character.yaml"))) {
                    new VoicebankConfig { SingerType = "utau" }.Save(stream);
                }
                foreach (string model in new[] { "t.bin", "p.bin", "s.bin", "v.bin" }) {
                    File.WriteAllBytes(Path.Combine(dir, "model", model), new byte[] { 0 });
                }

                var voicebank = new Voicebank();
                VoicebankLoader.LoadInfo(voicebank, characterPath, Path.GetDirectoryName(dir)!);

                Assert.Equal(USingerType.Neutrino, voicebank.SingerType);
            } finally {
                Directory.Delete(dir, recursive: true);
            }
        }
    }
}
