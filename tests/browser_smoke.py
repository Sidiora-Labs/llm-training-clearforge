"""Optional real-browser smoke check against a running Clearforge native-model service.

Run directly, outside pytest collection. Browser dependencies are deliberately optional.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    from playwright.sync_api import expect, sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("/tmp/clearforge-browser-check"))
    parser.add_argument("--api-key-env", default="CLEARFORGE_API_KEY")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"checks": [], "page_errors": [], "console_errors": []}
    with sync_playwright() as engine:
        browser = engine.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.set_default_timeout(30000)
        page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        page.on(
            "console",
            lambda message: report["console_errors"].append(message.text)
            if message.type == "error"
            else None,
        )
        try:
            page.goto(args.url, wait_until="networkidle")
            if key := os.environ.get(args.api_key_env):
                page.locator("#key").fill(key)
                page.locator("#connect").click()
            expect(page.locator("#model option")).not_to_have_count(0)
            expect(page.locator("#status")).to_contain_text("Ready")
            page.screenshot(path=str(args.output / "desktop.png"), full_page=True)
            report["checks"].append("model discovery and desktop layout")
            page.locator("#max").fill("16")
            page.locator("#temperature").fill("0")
            page.locator("#prompt").fill("Hi")
            page.locator("#send").click()
            expect(page.locator("#stop")).to_be_hidden()
            expect(page.locator("#error")).to_be_hidden()
            expect(page.locator(".message.assistant")).to_have_count(1)
            answer = page.locator(".message.assistant div").inner_text()
            page.screenshot(path=str(args.output / "conversation.png"), full_page=True)
            report["checks"].append("real model streamed answer")
            page.locator("#regenerate").click()
            expect(page.locator("#stop")).to_be_hidden()
            expect(page.locator(".message.user")).to_have_count(1)
            expect(page.locator(".message.assistant")).to_have_count(1)
            expect(page.locator(".message.assistant div")).to_have_text(answer)
            report["checks"].append("regenerate replaces the last answer")
            page.locator("#prompt").fill("More")
            page.locator("#prompt").press("Enter")
            expect(page.locator("#stop")).to_be_hidden()
            expect(page.locator("#error")).to_be_hidden()
            expect(page.locator(".message.user")).to_have_count(2)
            expect(page.locator(".message.assistant")).to_have_count(2)
            report["checks"].append("Enter and multi-turn history")
            page.locator("#reset").click()
            expect(page.locator(".message")).to_have_count(0)
            expect(page.locator("#welcome")).to_be_visible()
            expect(page.locator("#regenerate")).to_be_disabled()
            page.locator("[data-prompt]").first.click()
            assert page.locator("#prompt").input_value()
            report["checks"].append("reset restores welcome and working suggestions")
            page.locator("#max").fill("128")
            page.locator("#prompt").fill("Continue")
            page.locator("#send").click()
            expect(page.locator("#stop")).to_be_visible()
            page.locator("#stop").click()
            expect(page.locator("#stop")).to_be_hidden()
            expect(page.locator("#send")).to_be_enabled()
            report["checks"].append("Stop cancels the request and restores controls")
            page.wait_for_timeout(200)
            page.locator("#reset").click()
            page.locator("#max").fill("0")
            page.locator("#prompt").fill("Check input limits")
            page.locator("#send").click()
            expect(page.locator("#error")).to_be_visible()
            expect(page.locator("#send")).to_be_enabled()
            report["checks"].append("server validation error is visible")
            page.locator("#max").fill("16")
            page.locator("#regenerate").click()
            expect(page.locator("#stop")).to_be_hidden()
            expect(page.locator("#error")).to_be_hidden()
            expect(page.locator(".message.assistant")).to_have_count(1)
            report["checks"].append("regenerate retries after an error")
            options = page.locator("#model option").all()
            if len(options) > 1:
                page.locator("#model").select_option(options[1].get_attribute("value"))
                expect(page.locator(".message")).to_have_count(0)
                report["checks"].append("changing model clears conversation history")
            page.locator("#reset").click()
            page.set_viewport_size({"width": 390, "height": 844})
            expect(page.locator("#settings-panel")).to_be_hidden()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert (
                page.locator("#composer").bounding_box()["y"]
                + page.locator("#composer").bounding_box()["height"]
                <= 844
            )
            page.screenshot(path=str(args.output / "mobile.png"), full_page=True)
            page.locator("#settings-toggle").click()
            expect(page.locator("#settings-panel")).to_be_visible()
            expect(page.locator("#settings-toggle")).to_have_attribute("aria-expanded", "true")
            page.locator("#max").fill("16")
            page.screenshot(path=str(args.output / "mobile-settings.png"), full_page=True)
            page.locator("#settings-toggle").click()
            page.locator("#prompt").fill("Hi")
            page.locator("#send").click()
            expect(page.locator("#stop")).to_be_hidden()
            expect(page.locator("#error")).to_be_hidden()
            expect(page.locator(".message.assistant")).to_have_count(1)
            report["checks"].append("mobile settings, visible composer, and real generation")
            assert not report["page_errors"], report["page_errors"]
            # Browser network diagnostics for the deliberate 400 (and optional 401) are expected.
            unexpected = [
                error
                for error in report["console_errors"]
                if "status of 400" not in error and "status of 401" not in error
            ]
            assert not unexpected, unexpected
        except Exception:
            page.screenshot(path=str(args.output / "failure.png"), full_page=True)
            raise
        finally:
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            browser.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
