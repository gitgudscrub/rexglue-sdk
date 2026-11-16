/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <cstdlib>
#include <mutex>
#include <utility>

#include <rex/input/input.h>
#include <rex/input/input_system.h>
#include <rex/kernel/xam/private.h>
#include <rex/logging.h>
#include <rex/hook.h>
#include <rex/types.h>
#include <rex/cvar.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xtypes.h>

#include "xam_diag.h"

#pragma GCC diagnostic ignored "-Wunused-parameter"

namespace rex {
namespace kernel {
namespace xam {
using namespace rex::system;

using rex::input::X_INPUT_CAPABILITIES;
using rex::input::X_INPUT_KEYSTROKE;
using rex::input::X_INPUT_STATE;
using rex::input::X_INPUT_VIBRATION;

constexpr uint32_t XINPUT_FLAG_GAMEPAD = 0x01;
constexpr uint32_t XINPUT_FLAG_ANY_USER = 1 << 30;

REXCVAR_DEFINE_BOOL(primary_slot_remap, true, "Input",
                    "Front-end fix: a button press on any pad makes that pad player 1. "
                    "Turn off for co-op (players 2-4 must keep their own slots).");

namespace {

// Primary-slot remap (host convenience, not real-console behavior):
// whenever a button press STARTS on a controller slot, that slot becomes
// guest user 0 (swapped with whatever slot guest 0 currently maps to).
// This runtime hosts exactly one signed-in profile (user 0), so
// "whoever is pressing is player 1" keeps profile/XUID/signin queries
// valid whenever any of several plugged-in controllers drives the menu.
// Without it, a press from slot 1 makes the guest treat an anonymous
// user 1 as the primary player, which Fable 2's front-end cannot handle
// (black main menu). Press edges (bit 0 -> down) trigger the swap; held
// buttons do not, so the pads don't ping-pong while both are pressed.
struct PrimarySlotRemap {
  std::mutex lock;
  uint32_t map[4] = {0, 1, 2, 3};  // guest user -> host slot
  uint16_t last_buttons[4] = {0, 0, 0, 0};  // per host slot
};

PrimarySlotRemap& primary_slot_remap() {
  static PrimarySlotRemap remap;
  return remap;
}

// FABLE2_PRIMARY_SLOT_REMAP=0 disables the remap (A/B verification).
bool primary_slot_remap_enabled() {
  static const bool enabled = [] {
    char v[8] = {};
    size_t n = 0;
#if REX_PLATFORM_WIN32
    if (::getenv_s(&n, v, sizeof(v), "FABLE2_PRIMARY_SLOT_REMAP") != 0) return true;
#else
    const char* s = std::getenv("FABLE2_PRIMARY_SLOT_REMAP");
    if (!s) return true;
    v[0] = s[0];
#endif
    return v[0] != '0';
  }();
  return enabled;
}

// Maps a guest user index to the host XInput slot to query/report.
uint32_t host_slot_for_guest(u32 user_index) {
  auto& r = primary_slot_remap();
  if (user_index >= 4 || !REXCVAR_GET(primary_slot_remap)) {
    return user_index;
  }
  std::lock_guard<std::mutex> l(r.lock);
  return r.map[user_index];
}

// Call after a successful state read of host slot `host_slot` that was
// requested for guest user `user_index`. Swaps the pressed slot into
// guest user 0 when a press edge starts there. Returns true when the
// mapping changed, so the caller re-queries the now-swapped-in slot and
// hands the guest a consistent post-swap frame (the press registers at
// user 0 on the next poll instead of leaking in as a user-N press).
bool note_guest_button_press(u32 user_index, u32 host_slot, u16 buttons) {
  if (!primary_slot_remap_enabled() || user_index >= 4 || host_slot >= 4) {
    return false;
  }
  auto& r = primary_slot_remap();
  std::lock_guard<std::mutex> l(r.lock);
  if (!REXCVAR_GET(primary_slot_remap)) {
    // Off (co-op): put every user back on its own slot.
    for (uint32_t i = 0; i < 4; ++i) r.map[i] = i;
    r.last_buttons[host_slot] = buttons;
    return false;
  }
  const bool press_edge = (buttons & ~r.last_buttons[host_slot]) != 0;
  r.last_buttons[host_slot] = buttons;
  if (!press_edge || user_index == 0) {
    return false;
  }
  const uint32_t displaced = r.map[0];
  std::swap(r.map[0], r.map[user_index]);
  REXKRNL_INFO(
      "[XAM] primary-slot remap: press edge on host slot {} - "
      "guest user 0 now maps to host slot {} (was slot {})",
      (uint32_t)host_slot, (uint32_t)host_slot, (uint32_t)displaced);
  return true;
}

}  // namespace

rex::input::InputSystem* input_system() {
  return static_cast<rex::input::InputSystem*>(REX_KERNEL_STATE()->emulator()->input_system());
}

void XamResetInactivity_entry() {
  // Do we need to do anything?
}

u32 XamEnableInactivityProcessing_entry(u32 unk, u32 enable) {
  return X_ERROR_SUCCESS;
}

// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.reference.xinputgetcapabilities(v=vs.85).aspx
u32 XamInputGetCapabilities_entry(u32 user_index, u32 flags, ppc_ptr_t<X_INPUT_CAPABILITIES> caps) {
  REXKRNL_TRACE("[XAM] XamInputGetCapabilities called: user={}, flags=0x{:X}", (uint32_t)user_index,
                (uint32_t)flags);
  if (!caps) {
    return X_ERROR_BAD_ARGUMENTS;
  }

  if ((flags & 0xFF) && (flags & XINPUT_FLAG_GAMEPAD) == 0) {
    // Ignore any query for other types of devices.
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  uint32_t actual_user_index = user_index;
  if ((actual_user_index & 0xFF) == 0xFF || (flags & XINPUT_FLAG_ANY_USER)) {
    // Always pin user to 0.
    actual_user_index = 0;
  }
  const uint32_t host_slot = host_slot_for_guest(actual_user_index);

  auto* is = input_system();
  return is->GetCapabilities(host_slot, flags, caps);
}

u32 XamInputGetCapabilitiesEx_entry(u32 unk, u32 user_index, u32 flags,
                                    ppc_ptr_t<X_INPUT_CAPABILITIES> caps) {
  if (!caps) {
    return X_ERROR_BAD_ARGUMENTS;
  }

  if ((flags & 0xFF) && (flags & XINPUT_FLAG_GAMEPAD) == 0) {
    // Ignore any query for other types of devices.
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  uint32_t actual_user_index = user_index;
  if ((actual_user_index & 0xFF) == 0xFF || (flags & XINPUT_FLAG_ANY_USER)) {
    // Always pin user to 0.
    actual_user_index = 0;
  }

  (void)unk;  // Unused in this implementation
  const uint32_t host_slot = host_slot_for_guest(actual_user_index);
  auto* is = input_system();
  uint32_t result = is->GetCapabilities(host_slot, flags, caps);
  if (diag::enabled()) {
    char m[96];
    std::snprintf(m, sizeof(m), "XamInputGetCapabilities user=%u flags=0x%X -> 0x%X",
                  (unsigned)actual_user_index, (unsigned)flags, (unsigned)result);
    diag::trace(m);
  }
  return result;
}

// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.reference.xinputgetstate(v=vs.85).aspx
u32 XamInputGetState_entry(u32 user_index, u32 flags, ppc_ptr_t<X_INPUT_STATE> input_state) {
  // Games call this with a NULL state ptr, probably as a query.
  static int call_count = 0;
  if (++call_count <= 5) {
    REXKRNL_TRACE("[XAM] XamInputGetState called: user={}, flags=0x{:X}", (uint32_t)user_index,
                  (uint32_t)flags);
  }

  if ((flags & 0xFF) && (flags & XINPUT_FLAG_GAMEPAD) == 0) {
    // Ignore any query for other types of devices.
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  uint32_t actual_user_index = user_index;
  if ((actual_user_index & 0xFF) == 0xFF || (flags & XINPUT_FLAG_ANY_USER)) {
    // Always pin user to 0.
    actual_user_index = 0;
  }
  const uint32_t host_slot = host_slot_for_guest(actual_user_index);

  auto* is = input_system();
  uint32_t result = is->GetState(host_slot, input_state);
  if (result == 0 && input_state) {
    // A press edge on any slot makes that slot guest user 0 (primary remap).
    if (note_guest_button_press(actual_user_index, host_slot,
                                input_state->gamepad.buttons) &&
        input_state.guest_address() != 0) {
      // Serve the consistent post-swap state for this guest index instead of
      // the slot that just moved out from under it.
      const uint32_t new_slot = host_slot_for_guest(actual_user_index);
      if (new_slot != host_slot) {
        result = is->GetState(new_slot, input_state);
      }
    }
    if (diag::input_state_changed(actual_user_index, input_state->gamepad.buttons,
                                  input_state->gamepad.left_trigger,
                                  input_state->gamepad.right_trigger)) {
      char m[128];
      std::snprintf(m, sizeof(m),
                    "XamInputGetState user=%u (host slot %u) CHANGED buttons=0x%04X lt=%u rt=%u",
                    (unsigned)actual_user_index, (unsigned)host_slot,
                    (unsigned)input_state->gamepad.buttons,
                    (unsigned)input_state->gamepad.left_trigger,
                    (unsigned)input_state->gamepad.right_trigger);
      diag::trace(m);
    }
  }
  return result;
}

// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.reference.xinputsetstate(v=vs.85).aspx
u32 XamInputSetState_entry(u32 user_index, u32 unk, ppc_ptr_t<X_INPUT_VIBRATION> vibration) {
  if (!vibration) {
    return X_ERROR_BAD_ARGUMENTS;
  }

  uint32_t actual_user_index = user_index;
  if ((user_index & 0xFF) == 0xFF) {
    // Always pin user to 0.
    actual_user_index = 0;
  }
  const uint32_t host_slot = host_slot_for_guest(actual_user_index);

  (void)unk;  // Unused in this implementation
  auto* is = input_system();
  uint32_t result = is->SetState(host_slot, vibration);
  if (diag::enabled()) {
    char m[128];
    std::snprintf(m, sizeof(m), "XamInputSetState user=%u l=%u r=%u -> 0x%X",
                  (unsigned)actual_user_index,
                  (unsigned)vibration->left_motor_speed,
                  (unsigned)vibration->right_motor_speed, (unsigned)result);
    diag::trace(m);
  }
  return result;
}

// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.reference.xinputgetkeystroke(v=vs.85).aspx
u32 XamInputGetKeystroke_entry(u32 user_index, u32 flags, ppc_ptr_t<X_INPUT_KEYSTROKE> keystroke) {
  // https://github.com/CodeAsm/ffplay360/blob/master/Common/AtgXime.cpp
  // user index = index or XUSER_INDEX_ANY
  // flags = XINPUT_FLAG_GAMEPAD (| _ANYUSER | _ANYDEVICE)

  if (!keystroke) {
    return X_ERROR_BAD_ARGUMENTS;
  }

  if ((flags & 0xFF) && (flags & XINPUT_FLAG_GAMEPAD) == 0) {
    // Ignore any query for other types of devices.
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  uint32_t actual_user_index = user_index;
  if ((actual_user_index & 0xFF) == 0xFF || (flags & XINPUT_FLAG_ANY_USER)) {
    // Always pin user to 0.
    actual_user_index = 0;
  }
  const uint32_t host_slot = host_slot_for_guest(actual_user_index);

  auto* is = input_system();
  uint32_t result = is->GetKeystroke(host_slot, flags, keystroke);
  if (diag::enabled()) {
    char m[160];
    std::snprintf(m, sizeof(m), "XamInputGetKeystroke user=%u flags=0x%X -> 0x%X vk=%u uni=%u",
                  (unsigned)actual_user_index, (unsigned)flags, (unsigned)result,
                  (unsigned)keystroke->virtual_key, (unsigned)keystroke->unicode);
    diag::trace(m);
  }
  return result;
}

// Same as non-ex, just takes a pointer to user index.
u32 XamInputGetKeystrokeEx_entry(mapped_u32 user_index_ptr, u32 flags,
                                 ppc_ptr_t<X_INPUT_KEYSTROKE> keystroke) {
  if (!keystroke) {
    return X_ERROR_BAD_ARGUMENTS;
  }

  if ((flags & 0xFF) && (flags & XINPUT_FLAG_GAMEPAD) == 0) {
    // Ignore any query for other types of devices.
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  uint32_t user_index = *user_index_ptr;
  if ((user_index & 0xFF) == 0xFF || (flags & XINPUT_FLAG_ANY_USER)) {
    // Always pin user to 0.
    user_index = 0;
  }
  const uint32_t host_slot = host_slot_for_guest(user_index);

  auto* is = input_system();
  auto result = is->GetKeystroke(host_slot, flags, keystroke);
  if (XSUCCEEDED(result)) {
    *user_index_ptr = keystroke->user_index;
  }
  if (diag::enabled()) {
    char m[96];
    std::snprintf(m, sizeof(m), "XamInputGetKeystroke user=%u flags=0x%X -> 0x%X",
                  (unsigned)user_index, (unsigned)flags, (unsigned)result);
    diag::trace(m);
  }
  return result;
}

i32 XamUserGetDeviceContext_entry(u32 user_index, u32 unk, mapped_u32 out_ptr) {
  // Games check the result - usually with some masking.
  // If this function fails they assume zero, so let's fail AND
  // set zero just to be safe.
  *out_ptr = 0;
  uint32_t result = !user_index || (user_index & 0xFF) == 0xFF ? X_E_SUCCESS
                                                               : X_E_DEVICE_NOT_CONNECTED;
  if (diag::enabled()) {
    char m[96];
    std::snprintf(m, sizeof(m), "XamUserGetDeviceContext user=%u -> 0x%X",
                  (unsigned)user_index, (unsigned)result);
    diag::trace(m);
  }
  return result;
}

}  // namespace xam
}  // namespace kernel
}  // namespace rex

REX_EXPORT(__imp__XamResetInactivity, rex::kernel::xam::XamResetInactivity_entry)
REX_EXPORT(__imp__XamEnableInactivityProcessing,
           rex::kernel::xam::XamEnableInactivityProcessing_entry)
REX_EXPORT(__imp__XamInputGetCapabilities, rex::kernel::xam::XamInputGetCapabilities_entry)
REX_EXPORT(__imp__XamInputGetCapabilitiesEx, rex::kernel::xam::XamInputGetCapabilitiesEx_entry)
REX_EXPORT(__imp__XamInputGetState, rex::kernel::xam::XamInputGetState_entry)
REX_EXPORT(__imp__XamInputSetState, rex::kernel::xam::XamInputSetState_entry)
REX_EXPORT(__imp__XamInputGetKeystroke, rex::kernel::xam::XamInputGetKeystroke_entry)
REX_EXPORT(__imp__XamInputGetKeystrokeEx, rex::kernel::xam::XamInputGetKeystrokeEx_entry)
REX_EXPORT(__imp__XamUserGetDeviceContext, rex::kernel::xam::XamUserGetDeviceContext_entry)

REX_EXPORT_STUB(__imp__XamInputControl);
REX_EXPORT_STUB(__imp__XamInputEnableAutobind);
REX_EXPORT_STUB(__imp__XamInputGetDeviceStats);
REX_EXPORT_STUB(__imp__XamInputGetFailedConnectionOrBind);
REX_EXPORT_STUB(__imp__XamInputGetKeyLocks);
REX_EXPORT_STUB(__imp__XamInputGetKeystrokeHud);
REX_EXPORT_STUB(__imp__XamInputGetKeystrokeHudEx);
REX_EXPORT_STUB(__imp__XamInputGetUserVibrationLevel);
REX_EXPORT_STUB(__imp__XamInputNonControllerGetRaw);
REX_EXPORT_STUB(__imp__XamInputNonControllerGetRawEx);
REX_EXPORT_STUB(__imp__XamInputNonControllerSetRaw);
REX_EXPORT_STUB(__imp__XamInputNonControllerSetRawEx);
REX_EXPORT_STUB(__imp__XamInputRawState);
REX_EXPORT_STUB(__imp__XamInputResetLayoutKeyboard);
REX_EXPORT_STUB(__imp__XamInputSendStayAliveRequest);
REX_EXPORT_STUB(__imp__XamInputSendXenonButtonPress);
REX_EXPORT_STUB(__imp__XamInputSetKeyLocks);
REX_EXPORT_STUB(__imp__XamInputSetKeyboardTranslationHud);
REX_EXPORT_STUB(__imp__XamInputSetLayoutKeyboard);
REX_EXPORT_STUB(__imp__XamInputSetMinMaxAuthDelay);
REX_EXPORT_STUB(__imp__XamInputSetTextMessengerIndicator);
REX_EXPORT_STUB(__imp__XamInputToggleKeyLocks);
