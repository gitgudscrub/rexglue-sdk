/**
 * @file        ui/overlay/notice.h
 * @brief       A short on-screen notice (title, text, small footer) drawn at the
 *              top of the window over the game, for titles that cannot use
 *              ImGui themselves (it is not exported from the runtime).
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */
#pragma once

#include <functional>
#include <string>

namespace rex::ui {

class ImGuiDrawer;

enum class NoticeKind { kInfo, kWarning };

// Shows a notice for `seconds` (fades in and out), replacing any current one.
// Thread-safe: callable from any thread. Takes no input.
void ShowNotice(NoticeKind kind, std::string title, std::string body, std::string footer,
                float seconds);
void HideNotice();

// Runtime side: ReXApp hands over its ImGui drawer and a way to run code on
// the UI thread (dialogs are made there, when a notice is shown).
void InstallNoticeOverlay(ImGuiDrawer* drawer, std::function<void(std::function<void()>)> post_ui);

}  // namespace rex::ui
