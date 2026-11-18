/**
 * @file        ui/overlay/notice.cpp
 * @brief       On-screen notice. See notice.h.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */
#include <rex/ui/overlay/notice.h>

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>

#include <rex/ui/imgui_dialog.h>
#include <rex/logging.h>
#include <rex/ui/imgui_drawer.h>

namespace rex::ui {
namespace {

using Clock = std::chrono::steady_clock;

struct Notice {
  NoticeKind kind = NoticeKind::kInfo;
  std::string title, body, footer;
  float seconds = 0.0f;
  Clock::time_point shown{};
  bool active = false;
};

// The dialog exists only while a notice shows (an ImGui dialog keeps the ImGui
// layer drawing and its input listener attached), so it is made on demand on
// the UI thread and closes itself when the notice ends.
struct Overlay {
  ImGuiDrawer* drawer = nullptr;
  std::function<void(std::function<void()>)> post_ui;
  bool dialog_alive = false;
};
Overlay& overlay() {
  static Overlay o;
  return o;
}

std::mutex& notice_mutex() {
  static std::mutex m;
  return m;
}
Notice& current_notice() {
  static Notice n;
  return n;
}

class NoticeDialog : public ImGuiDialog {
 public:
  explicit NoticeDialog(ImGuiDrawer* drawer) : ImGuiDialog(drawer) {}

 protected:
  void OnDraw(ImGuiIO& io) override {
    Notice n;
    {
      std::lock_guard<std::mutex> lock(notice_mutex());
      Notice& cur = current_notice();
      if (cur.active &&
          std::chrono::duration<float>(Clock::now() - cur.shown).count() >= cur.seconds)
        cur.active = false;
      if (!cur.active) {
        overlay().dialog_alive = false;  // a later ShowNotice makes a new dialog
        Close();
        REXLOG_INFO("[notice] closed after {} frames", frames_);
        return;
      }
      n = cur;
    }
    if (n.shown != logged_) {
      logged_ = n.shown;
      REXLOG_INFO("[notice] showing: {}", n.title);
    }
    ++frames_;
    const float age = std::chrono::duration<float>(Clock::now() - n.shown).count();
    float alpha = std::min(1.0f, age / 0.3f);
    if (age > n.seconds - 0.6f) alpha = std::max(0.0f, (n.seconds - age) / 0.6f);

    const float scale = std::max(1.0f, io.DisplaySize.y / 1080.0f);
    const float w = std::min(io.DisplaySize.x - 32.0f, 560.0f * scale);
    ImGui::SetNextWindowPos(ImVec2((io.DisplaySize.x - w) * 0.5f, 24.0f * scale),
                            ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(w, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.88f * alpha);
    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_AlwaysAutoResize;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f * scale);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f * scale, 10.0f * scale));
    if (ImGui::Begin("##rex_notice", nullptr, flags)) {
      ImGui::SetWindowFontScale(1.15f * scale);
      const ImVec4 title_color = n.kind == NoticeKind::kWarning
                                     ? ImVec4(1.0f, 0.72f, 0.35f, alpha)
                                     : ImVec4(0.55f, 0.9f, 0.55f, alpha);
      ImGui::PushStyleColor(ImGuiCol_Text, title_color);
      ImGui::TextUnformatted(n.title.c_str());
      ImGui::PopStyleColor();
      if (!n.body.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.95f, 0.95f, alpha));
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(n.body.c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
      }
      if (!n.footer.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, alpha));
        ImGui::TextUnformatted(n.footer.c_str());
        ImGui::PopStyleColor();
      }
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
  }

 private:
  Clock::time_point logged_{};
  uint64_t frames_ = 0;
};

}  // namespace

void ShowNotice(NoticeKind kind, std::string title, std::string body, std::string footer,
                float seconds) {
  ImGuiDrawer* make_on = nullptr;
  std::function<void(std::function<void()>)> post;
  {
    std::lock_guard<std::mutex> lock(notice_mutex());
    Notice& n = current_notice();
    n.kind = kind;
    n.title = std::move(title);
    n.body = std::move(body);
    n.footer = std::move(footer);
    n.seconds = std::clamp(seconds, 1.0f, 120.0f);
    n.shown = Clock::now();
    n.active = true;
    Overlay& o = overlay();
    if (!o.dialog_alive && o.drawer && o.post_ui) {
      o.dialog_alive = true;
      make_on = o.drawer;
      post = o.post_ui;
    }
  }
  // outside the lock: on the UI thread this may run at once
  if (make_on) post([make_on] { (void)new NoticeDialog(make_on); });  // retains itself
}

void HideNotice() {
  std::lock_guard<std::mutex> lock(notice_mutex());
  current_notice().active = false;
}

void InstallNoticeOverlay(ImGuiDrawer* drawer, std::function<void(std::function<void()>)> post_ui) {
  bool show_now = false;
  std::function<void(std::function<void()>)> post;
  {
    std::lock_guard<std::mutex> lock(notice_mutex());
    overlay().drawer = drawer;
    overlay().post_ui = std::move(post_ui);
    post = overlay().post_ui;
    show_now = current_notice().active && !overlay().dialog_alive && drawer && post;
    if (show_now) overlay().dialog_alive = true;
  }
  if (show_now) post([drawer] { (void)new NoticeDialog(drawer); });
}

}  // namespace rex::ui
