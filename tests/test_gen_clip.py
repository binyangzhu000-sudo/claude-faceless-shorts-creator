import importlib.util
import io
import pathlib
import unittest
from unittest import mock
import urllib.error


MODULE_PATH = pathlib.Path(__file__).parents[1] / "tools" / "gen_clip.py"
SPEC = importlib.util.spec_from_file_location("gen_clip", MODULE_PATH)
gen_clip = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen_clip)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AtlasProviderTests(unittest.TestCase):
    def test_payload_maps_established_frame_aliases(self):
        payload = gen_clip.atlas_payload({
            "image_url": "https://example.com/start.png",
            "end_image_url": "https://example.com/end.png",
            "duration": "5",
        })

        self.assertEqual(payload["image"], "https://example.com/start.png")
        self.assertEqual(payload["last_image"], "https://example.com/end.png")
        self.assertEqual(payload["duration"], 5)
        self.assertNotIn("image_url", payload)
        self.assertNotIn("end_image_url", payload)

    @mock.patch.object(gen_clip.urllib.request, "urlopen")
    def test_submit_failure_is_not_retried(self, urlopen):
        urlopen.side_effect = urllib.error.URLError("temporary failure")

        with self.assertRaises(gen_clip.AtlasApiError):
            gen_clip.atlas_json("https://example.com/submit", "secret", body={"model": "m"},
                                sleep=lambda _: None)

        self.assertEqual(urlopen.call_count, 1)

    @mock.patch.object(gen_clip.urllib.request, "urlopen")
    def test_result_get_retries_transient_failure(self, urlopen):
        urlopen.side_effect = [
            urllib.error.URLError("temporary failure"),
            FakeResponse(b'{"code":200,"data":{"status":"processing"}}'),
        ]
        sleeps = []

        result = gen_clip.atlas_json("https://example.com/result/1", "secret",
                                     sleep=sleeps.append)

        self.assertEqual(result["data"]["status"], "processing")
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(sleeps, [1])

    @mock.patch.object(gen_clip.urllib.request, "urlopen")
    def test_result_get_respects_deadline(self, urlopen):
        urlopen.side_effect = urllib.error.URLError("slow")
        sleeps = []
        deadline = gen_clip.time.monotonic() + 3

        with self.assertRaises(gen_clip.AtlasApiError):
            gen_clip.atlas_json("https://example.com/p/1", "secret",
                                sleep=sleeps.append, deadline=deadline)

        for call in urlopen.call_args_list:
            self.assertLessEqual(call.kwargs["timeout"], 3)
        self.assertTrue(all(s <= 3 for s in sleeps))

    def test_expired_deadline_makes_no_request(self):
        with mock.patch.object(gen_clip.urllib.request, "urlopen") as urlopen:
            with self.assertRaises(gen_clip.AtlasApiError):
                gen_clip.atlas_json("https://example.com/p/1", "secret",
                                    deadline=gen_clip.time.monotonic() - 1)
        urlopen.assert_not_called()

    @mock.patch.object(gen_clip, "atlas_json")
    def test_completed_prediction_returns_first_output(self, atlas_json):
        atlas_json.side_effect = [
            {"data": {"id": "prediction-1", "status": "starting"}},
            {"data": {"id": "prediction-1", "status": "completed",
                      "outputs": ["https://cdn.example.com/video.mp4"]}},
        ]

        prediction_id, url = gen_clip.run_atlas("model", {"prompt": "test"}, "secret", 10)

        self.assertEqual(prediction_id, "prediction-1")
        self.assertEqual(url, "https://cdn.example.com/video.mp4")
        self.assertEqual(atlas_json.call_count, 2)
        self.assertIsNotNone(atlas_json.call_args_list[0].kwargs["body"])
        self.assertNotIn("body", atlas_json.call_args_list[1].kwargs)
        self.assertTrue(atlas_json.call_args_list[1].args[0].endswith("/model/prediction/prediction-1"))


if __name__ == "__main__":
    unittest.main()
