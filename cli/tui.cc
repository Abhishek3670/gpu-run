#include "cli/tui.h"

// The wizard keeps one ncurses session alive from entry through the optional
// log viewer so the terminal is initialized and restored exactly once.
// Screen collectors only gather input; RPCs stay in RunWizard or viewer threads.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <ncurses.h>
#include <unistd.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "cli/client.h"

namespace gpu_run::tui {
namespace {

constexpr int kMinLines = 20;
constexpr int kMinCols = 60;
constexpr int kHeaderColorPair = 1;
constexpr int kOkColorPair = 2;
constexpr int kValueColorPair = 3;
constexpr int kErrorColorPair = 4;
constexpr int kStatusBarColorPair = 5;
constexpr int kHintColorPair = 6;
constexpr int kEscapeKey = 27;
constexpr int kTextInputMax = 255;
constexpr std::size_t kMaxLogLines = 5000;
constexpr std::chrono::seconds kStatusPollInterval(2);

std::atomic<bool> g_interrupted(false);
std::string g_task_type_label = "training";

enum class GpuSelectScreenResult {
  Confirm,
  Quit,
  Refresh,
};

bool IsEnterKey(int ch) {
  return ch == '\n' || ch == '\r' || ch == KEY_ENTER || ch == 10 || ch == 13;
}

bool IsQuitKey(int ch) {
  return ch == 'q' || ch == 'Q' || ch == kEscapeKey;
}

void SigIntHandler(int) {
  g_interrupted.store(true);
  endwin();
  _exit(0);
}

std::string EscapeJson(std::string_view input) {
  std::string output;
  output.reserve(input.size());
  for (const char ch : input) {
    switch (ch) {
      case '\\':
        output += "\\\\";
        break;
      case '"':
        output += "\\\"";
        break;
      case '\n':
        output += "\\n";
        break;
      default:
        output.push_back(ch);
        break;
    }
  }
  return output;
}

std::string FormatTaskType(gpu::TaskType task_type) {
  return task_type == gpu::TaskType::COMPUTE ? "compute" : "training";
}

std::string FormatPriority(gpu::Priority priority) {
  switch (priority) {
    case gpu::Priority::LOW:
      return "low";
    case gpu::Priority::HIGH:
      return "high";
    case gpu::Priority::MEDIUM:
    case gpu::Priority::PRIORITY_UNSPECIFIED:
      return "medium";
  }
  return "medium";
}

std::string FormatGpuIdList(const std::vector<int>& gpu_ids) {
  if (gpu_ids.empty()) {
    return "none";
  }

  std::ostringstream stream;
  for (std::size_t index = 0; index < gpu_ids.size(); ++index) {
    if (index != 0) {
      stream << ", ";
    }
    stream << "GPU " << gpu_ids[index];
  }
  return stream.str();
}

std::string FormatMemoryGiB(std::uint64_t bytes) {
  const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(1) << gib;
  return stream.str();
}

std::string FormatElapsed(std::chrono::steady_clock::duration elapsed) {
  const auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
  const long long hours = total_seconds / 3600;
  const long long minutes = (total_seconds % 3600) / 60;
  const long long seconds = total_seconds % 60;

  std::ostringstream stream;
  stream << std::setfill('0') << std::setw(2) << hours << ':' << std::setw(2) << minutes << ':'
         << std::setw(2) << seconds;
  return stream.str();
}

bool IsTerminalState(std::string_view state) {
  return state == "SUCCEEDED" || state == "FAILED" || state == "CANCELED";
}

int StateBadgeAttributes(std::string_view state) {
  if (state == "QUEUED") {
    return COLOR_PAIR(kHeaderColorPair) | A_DIM;
  }
  if (state == "DISPATCHING") {
    return COLOR_PAIR(kValueColorPair);
  }
  if (state == "RUNNING") {
    return COLOR_PAIR(kOkColorPair);
  }
  if (state == "SUCCEEDED") {
    return COLOR_PAIR(kOkColorPair) | A_BOLD;
  }
  if (state == "FAILED") {
    return COLOR_PAIR(kErrorColorPair) | A_BOLD;
  }
  if (state == "CANCELED") {
    return COLOR_PAIR(kValueColorPair) | A_DIM;
  }
  return COLOR_PAIR(kErrorColorPair) | A_BOLD;
}

gpu::cli::ClientOptions ToClientOptions(const JobConfig& config) {
  gpu::cli::ClientOptions options;
  options.server_address = config.server_addr;
  if (!config.token.empty()) {
    options.bearer_token = config.token;
  }
  return options;
}

std::vector<GpuInfo> ToGpuInfos(const std::vector<gpu::cli::GpuInfoView>& views) {
  std::vector<GpuInfo> gpus;
  gpus.reserve(views.size());
  for (const auto& view : views) {
    GpuInfo gpu;
    gpu.gpu_id = view.gpu_id;
    gpu.model_name = view.model_name;
    gpu.total_memory_bytes = view.total_memory_bytes;
    gpu.used_memory_bytes = view.used_memory_bytes;
    gpu.utilization_percent = view.utilization_percent;
    gpu.available = view.available;
    gpu.locked_job_id = view.locked_job_id;
    gpus.push_back(std::move(gpu));
  }
  return gpus;
}

void PrintPaddedLine(int row, const std::string& text, int attributes) {
  if (row < 0 || row >= LINES || COLS <= 0) {
    return;
  }

  std::string line = text;
  if (static_cast<int>(line.size()) < COLS) {
    line.append(static_cast<std::size_t>(COLS - static_cast<int>(line.size())), ' ');
  } else if (static_cast<int>(line.size()) > COLS) {
    line.resize(static_cast<std::size_t>(COLS));
  }

  attron(attributes);
  mvaddnstr(row, 0, line.c_str(), COLS);
  attroff(attributes);
}

void PrintClipped(int row, int col, int width, const std::string& text, int attributes) {
  if (row < 0 || row >= LINES || col < 0 || col >= COLS || width <= 0) {
    return;
  }

  const int bounded_width = std::min(width, COLS - col);
  if (bounded_width <= 0) {
    return;
  }

  attron(attributes);
  mvaddnstr(row, col, text.c_str(), bounded_width);
  attroff(attributes);
}

void DrawHeader(const std::string& title) {
  PrintPaddedLine(0, title, COLOR_PAIR(kHeaderColorPair) | A_BOLD);
}

void DrawHint(const std::string& hint) {
  PrintPaddedLine(LINES - 1, hint, COLOR_PAIR(kHintColorPair) | A_DIM);
}

void DrawBox(int top, int left, int height, int width, int color_pair) {
  if (height < 2 || width < 2) {
    return;
  }

  attron(COLOR_PAIR(color_pair));
  mvaddch(top, left, ACS_ULCORNER);
  mvaddch(top, left + width - 1, ACS_URCORNER);
  mvaddch(top + height - 1, left, ACS_LLCORNER);
  mvaddch(top + height - 1, left + width - 1, ACS_LRCORNER);
  for (int col = left + 1; col < left + width - 1; ++col) {
    mvaddch(top, col, ACS_HLINE);
    mvaddch(top + height - 1, col, ACS_HLINE);
  }
  for (int row = top + 1; row < top + height - 1; ++row) {
    mvaddch(row, left, ACS_VLINE);
    mvaddch(row, left + width - 1, ACS_VLINE);
  }
  attroff(COLOR_PAIR(color_pair));
}

std::vector<std::string> WrapText(const std::string& text, int width) {
  if (width <= 0) {
    return {text};
  }

  std::vector<std::string> lines;
  std::istringstream stream(text);
  std::string word;
  std::string current;
  while (stream >> word) {
    if (current.empty()) {
      current = word;
    } else if (static_cast<int>(current.size() + 1 + word.size()) <= width) {
      current += ' ';
      current += word;
    } else {
      lines.push_back(current);
      current = word;
    }
  }
  if (!current.empty()) {
    lines.push_back(current);
  }
  if (lines.empty()) {
    lines.push_back(text.substr(0, static_cast<std::size_t>(width)));
  }
  return lines;
}

void FlashMessage(const std::string& message, int color_pair) {
  PrintPaddedLine(LINES - 2, message, COLOR_PAIR(color_pair) | A_BOLD);
  refresh();
  napms(900);
}

std::string PromptTextInput(int row, int col, int width, const std::string& current_value, bool keep_if_empty) {
  char buffer[kTextInputMax + 1] = {};
  const int max_chars = std::max(1, std::min(kTextInputMax, width - 1));

  move(row, col);
  clrtoeol();
  refresh();
  echo();
  curs_set(1);
  const int result = getnstr(buffer, max_chars);
  noecho();
  curs_set(0);

  if (result == ERR) {
    return current_value;
  }

  const std::string input(buffer);
  if (input.empty() && keep_if_empty) {
    return current_value;
  }
  return input;
}

int FirstSelectableGpuIndex(const std::vector<GpuInfo>& gpus) {
  for (std::size_t index = 0; index < gpus.size(); ++index) {
    if (gpus[index].available) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int NextSelectableGpuIndex(const std::vector<GpuInfo>& gpus, int current_index, int direction) {
  if (gpus.empty()) {
    return -1;
  }

  int index = current_index;
  for (std::size_t attempt = 0; attempt < gpus.size(); ++attempt) {
    index += direction;
    if (index < 0) {
      index = static_cast<int>(gpus.size()) - 1;
    }
    if (index >= static_cast<int>(gpus.size())) {
      index = 0;
    }
    if (gpus[static_cast<std::size_t>(index)].available) {
      return index;
    }
  }
  return current_index;
}

std::vector<int> SelectedGpuIds(const std::vector<GpuInfo>& gpus, const std::vector<bool>& selected_rows) {
  std::vector<int> gpu_ids;
  for (std::size_t index = 0; index < gpus.size(); ++index) {
    if (gpus[index].available && selected_rows[index]) {
      gpu_ids.push_back(gpus[index].gpu_id);
    }
  }
  return gpu_ids;
}
bool ShowRetryQuitPanel(const std::string& message) {
  const std::vector<std::string> wrapped = WrapText(message, std::max(10, COLS - 18));
  const int panel_width = std::min(COLS - 8, std::max(40, COLS - 12));
  const int panel_height = std::min(LINES - 6, std::max(8, static_cast<int>(wrapped.size()) + 6));
  const int panel_top = std::max(2, (LINES - panel_height) / 2);
  const int panel_left = std::max(2, (COLS - panel_width) / 2);
  int selection = 0;

  while (!g_interrupted.load()) {
    erase();
    refresh();
    DrawBox(panel_top, panel_left, panel_height, panel_width, kErrorColorPair);
    PrintClipped(panel_top + 1, panel_left + 2, panel_width - 4, "error", COLOR_PAIR(kErrorColorPair) | A_BOLD);
    for (std::size_t line_index = 0; line_index < wrapped.size(); ++line_index) {
      PrintClipped(
          panel_top + 2 + static_cast<int>(line_index),
          panel_left + 2,
          panel_width - 4,
          wrapped[line_index],
          COLOR_PAIR(kErrorColorPair));
    }

    const std::vector<std::string> items = {"Retry", "Quit wizard"};
    for (int item_index = 0; item_index < static_cast<int>(items.size()); ++item_index) {
      const int attributes = item_index == selection ? COLOR_PAIR(kOkColorPair) | A_REVERSE : COLOR_PAIR(kValueColorPair);
      PrintClipped(
          panel_top + panel_height - 3 + item_index,
          panel_left + 4,
          panel_width - 8,
          items[static_cast<std::size_t>(item_index)],
          attributes);
    }
    DrawHint("arrow keys navigate   enter select   q quit wizard");
    refresh();

    const int ch = getch();
    if (IsQuitKey(ch)) {
      return false;
    }
    if (ch == KEY_UP || ch == KEY_LEFT) {
      selection = selection == 0 ? 1 : 0;
    } else if (ch == KEY_DOWN || ch == KEY_RIGHT) {
      selection = selection == 0 ? 1 : 0;
    } else if (IsEnterKey(ch)) {
      return selection == 0;
    }
  }

  return false;
}

void DrawUploadProgress(std::int64_t bytes_sent) {
  std::ostringstream stream;
  stream << "uploading... " << (bytes_sent / 1024) << " KB";

  erase();
  refresh();
  DrawHeader("gpu-run upload");
  PrintClipped(3, 2, COLS - 4, stream.str(), COLOR_PAIR(kValueColorPair) | A_BOLD);
  DrawHint("uploading bundle to server");
  refresh();
}

PostSubmitAction ShowPostSubmitActionMenu(const std::string& job_id) {
  const std::vector<std::string> items = {
      "Stream logs live",
      "Poll status until done",
      "Exit - print job ID",
  };
  int selection = 0;

  while (!g_interrupted.load()) {
    erase();
    refresh();
    DrawHeader("gpu-run submit");
    PrintClipped(3, 2, COLS - 4, "submitted: " + job_id, COLOR_PAIR(kOkColorPair) | A_BOLD);
    for (int index = 0; index < static_cast<int>(items.size()); ++index) {
      const int attributes = index == selection ? COLOR_PAIR(kOkColorPair) | A_REVERSE : COLOR_PAIR(kValueColorPair);
      PrintClipped(6 + index, 4, COLS - 8, items[static_cast<std::size_t>(index)], attributes);
    }
    DrawHint("arrow keys navigate   enter select   q exit viewer");
    refresh();

    const int ch = getch();
    if (IsQuitKey(ch)) {
      return PostSubmitAction::Exit;
    }
    if (ch == KEY_UP) {
      selection = selection == 0 ? static_cast<int>(items.size()) - 1 : selection - 1;
    } else if (ch == KEY_DOWN) {
      selection = selection == static_cast<int>(items.size()) - 1 ? 0 : selection + 1;
    } else if (IsEnterKey(ch)) {
      if (selection == 0) {
        return PostSubmitAction::StreamLogs;
      }
      if (selection == 1) {
        return PostSubmitAction::PollStatus;
      }
      return PostSubmitAction::Exit;
    }
  }

  return PostSubmitAction::Exit;
}

std::string BuildJobConfigSummaryLine(const std::string& label, const std::string& value) {
  std::ostringstream stream;
  stream << std::left << std::setw(14) << label << value;
  return stream.str();
}

GpuSelectScreenResult ShowGpuSelectScreen(JobConfig& config, const std::vector<GpuInfo>& gpus) {
  std::vector<bool> selected_rows(gpus.size(), false);
  for (std::size_t index = 0; index < gpus.size(); ++index) {
    if (gpus[index].available &&
        std::find(config.preferred_gpu_ids.begin(), config.preferred_gpu_ids.end(), gpus[index].gpu_id) !=
            config.preferred_gpu_ids.end()) {
      selected_rows[index] = true;
    }
  }

  int current_index = FirstSelectableGpuIndex(gpus);
  while (!g_interrupted.load()) {
    erase();
    refresh();
    DrawHeader("gpu-run GPU selection");
    PrintClipped(2, 2, COLS - 4, "[sel] GPU-N | name | util% | mem used/total GB | status", COLOR_PAIR(kHeaderColorPair));

    const int table_rows = std::max(1, LINES - 7);
    const int visible_count = std::min(static_cast<int>(gpus.size()), table_rows);
    for (int row_index = 0; row_index < visible_count; ++row_index) {
      const GpuInfo& gpu = gpus[static_cast<std::size_t>(row_index)];
      const bool selected = selected_rows[static_cast<std::size_t>(row_index)];
      int row_attributes = 0;
      if (!gpu.available) {
        row_attributes = COLOR_PAIR(kErrorColorPair) | A_DIM;
      } else if (selected) {
        row_attributes = COLOR_PAIR(kOkColorPair) | A_REVERSE | (current_index == row_index ? A_BOLD : 0);
      } else if (current_index == row_index) {
        row_attributes = COLOR_PAIR(kValueColorPair) | A_REVERSE;
      }

      std::ostringstream row;
      row << '[' << (selected ? 'x' : ' ') << "] GPU-" << gpu.gpu_id << " | " << gpu.model_name << " | "
          << gpu.utilization_percent << "% | " << FormatMemoryGiB(gpu.used_memory_bytes) << '/'
          << FormatMemoryGiB(gpu.total_memory_bytes) << " GB | ";
      PrintClipped(3 + row_index, 2, COLS - 4, row.str(), row_attributes);

      const std::string status_badge = gpu.available ? "free" : "busy " + gpu.locked_job_id;
      const int status_width = static_cast<int>(status_badge.size()) + 1;
      const int status_x = std::max(2, COLS - status_width - 2);
      const int status_attributes = gpu.available ? COLOR_PAIR(kOkColorPair) : COLOR_PAIR(kErrorColorPair) | A_DIM;
      PrintClipped(3 + row_index, status_x, COLS - status_x - 1, status_badge, status_attributes);
    }

    const std::vector<int> selected_gpu_ids = SelectedGpuIds(gpus, selected_rows);
    const std::string summary = "count: " + std::to_string(selected_gpu_ids.size()) + "   preferred: " +
        FormatGpuIdList(selected_gpu_ids);
    PrintClipped(std::min(LINES - 3, 4 + visible_count), 2, COLS - 4, summary, COLOR_PAIR(kValueColorPair));
    DrawHint("arrow keys navigate   space toggle   r refresh   enter confirm   q quit wizard");
    refresh();

    const int ch = getch();
    if (IsQuitKey(ch)) {
      return GpuSelectScreenResult::Quit;
    }
    if (ch == 'r' || ch == 'R') {
      return GpuSelectScreenResult::Refresh;
    }
    if (ch == KEY_UP && current_index >= 0) {
      current_index = NextSelectableGpuIndex(gpus, current_index, -1);
    } else if (ch == KEY_DOWN && current_index >= 0) {
      current_index = NextSelectableGpuIndex(gpus, current_index, 1);
    } else if (ch == ' ' && current_index >= 0 && gpus[static_cast<std::size_t>(current_index)].available) {
      selected_rows[static_cast<std::size_t>(current_index)] = !selected_rows[static_cast<std::size_t>(current_index)];
    } else if (IsEnterKey(ch)) {
      if (selected_gpu_ids.empty()) {
        FlashMessage("select at least one free GPU", kErrorColorPair);
        continue;
      }
      config.preferred_gpu_ids = selected_gpu_ids;
      config.gpu_count = static_cast<int>(selected_gpu_ids.size());
      return GpuSelectScreenResult::Confirm;
    }
  }

  return GpuSelectScreenResult::Quit;
}

void PrintStatusJson(const gpu::cli::JobStatusView& status) {
  std::cout << "{\"job_id\":\"" << EscapeJson(status.job_id)
            << "\",\"state\":\"" << EscapeJson(status.state)
            << "\",\"queue_position\":" << status.queue_position
            << ",\"exit_code\":" << status.exit_code
            << ",\"status_message\":\"" << EscapeJson(status.status_message)
            << "\",\"assigned_gpu_ids\":[";
  for (std::size_t gpu_index = 0; gpu_index < status.assigned_gpu_ids.size(); ++gpu_index) {
    std::cout << status.assigned_gpu_ids[gpu_index];
    if (gpu_index + 1 != status.assigned_gpu_ids.size()) {
      std::cout << ',';
    }
  }
  std::cout << "]}\n";
}
class LogViewerSession {
 public:
  LogViewerSession(std::string job_id, gpu::cli::GpuRunClient& client)
      : job_id_(std::move(job_id)), client_(client), started_at_(std::chrono::steady_clock::now()) {}

  void Run() {
    stop_.store(false);
    done_.store(false);

    log_thread_ = std::thread(&LogViewerSession::RunLogThread, this);
    status_thread_ = std::thread(&LogViewerSession::RunStatusThread, this);

    halfdelay(1);
    bool saw_done = false;
    while (!g_interrupted.load()) {
      Draw();
      if (done_.load()) {
        if (saw_done) {
          break;
        }
        saw_done = true;
      } else {
        saw_done = false;
      }

      const int ch = getch();
      if (ch == ERR) {
        continue;
      }
      if (HandleKey(ch)) {
        break;
      }
    }

    StopAndJoinThreads();
    cbreak();
  }

 private:
  void RunLogThread() {
    const absl::Status status = client_.StreamLogsCallback(job_id_, [this](std::string line) {
      if (stop_.load() || g_interrupted.load()) {
        return;
      }
      std::lock_guard<std::mutex> lock(ring_mutex_);
      AppendLogTextLocked(line);
    });

    std::lock_guard<std::mutex> lock(ring_mutex_);
    if (!pending_line_.empty()) {
      PushRingLineLocked(pending_line_);
      pending_line_.clear();
    }
    if (!status.ok() && !stop_.load() && status.code() != absl::StatusCode::kCancelled) {
      PushRingLineLocked("stream error: " + std::string(status.message()));
    }
    done_.store(true);
  }

  void RunStatusThread() {
    while (!stop_.load() && !g_interrupted.load()) {
      const auto status = client_.GetStatus(job_id_);
      {
        std::lock_guard<std::mutex> lock(status_mutex_);
        if (status.ok()) {
          state_str_ = status->state;
          status_str_ = "GPUs: " + FormatGpuIdList(status->assigned_gpu_ids);
        } else {
          state_str_ = "ERROR";
          status_str_ = "status error: " + std::string(status.status().message());
        }
      }
      if (status.ok() && IsTerminalState(status->state)) {
        done_.store(true);
      }

      for (int tick = 0; tick < 20; ++tick) {
        if (stop_.load() || g_interrupted.load()) {
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  }

  void AppendLogTextLocked(const std::string& text) {
    pending_line_.append(text);
    std::size_t newline_pos = pending_line_.find('\n');
    while (newline_pos != std::string::npos) {
      std::string line = pending_line_.substr(0, newline_pos);
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      PushRingLineLocked(line);
      pending_line_.erase(0, newline_pos + 1);
      newline_pos = pending_line_.find('\n');
    }
  }

  void PushRingLineLocked(const std::string& line) {
    ring_buffer_.push_back(line);
    while (ring_buffer_.size() > kMaxLogLines) {
      ring_buffer_.pop_front();
    }
  }

  std::size_t RingLineCount() const {
    std::lock_guard<std::mutex> lock(ring_mutex_);
    return ring_buffer_.size();
  }

  int MaxScrollOffset() const {
    const int visible_rows = std::max(1, LINES - 3);
    const int total_rows = static_cast<int>(RingLineCount());
    return std::max(0, total_rows - visible_rows);
  }

  void Draw() {
    erase();
    refresh();

    DrawHeader("gpu-run logs   " + job_id_);

    std::string state;
    std::string status;
    {
      std::lock_guard<std::mutex> lock(status_mutex_);
      state = state_str_;
      status = status_str_;
    }

    std::string status_line = " " + state + " | " + status + " | task: " + g_task_type_label +
        " | elapsed " + FormatElapsed(std::chrono::steady_clock::now() - started_at_);
    PrintPaddedLine(1, status_line, COLOR_PAIR(kStatusBarColorPair) | A_REVERSE);
    PrintClipped(1, 0, COLS, " " + state + " ", StateBadgeAttributes(state) | A_REVERSE);

    std::vector<std::string> lines;
    {
      std::lock_guard<std::mutex> lock(ring_mutex_);
      lines.assign(ring_buffer_.begin(), ring_buffer_.end());
    }

    const int visible_rows = std::max(1, LINES - 3);
    if (auto_scroll_) {
      scroll_offset_ = 0;
    }
    scroll_offset_ = std::clamp(scroll_offset_, 0, std::max(0, static_cast<int>(lines.size()) - visible_rows));
    const int start_index = std::max(0, static_cast<int>(lines.size()) - visible_rows - scroll_offset_);
    for (int row = 0; row < visible_rows; ++row) {
      const int line_index = start_index + row;
      std::string text;
      if (line_index >= 0 && line_index < static_cast<int>(lines.size())) {
        text = lines[static_cast<std::size_t>(line_index)];
      }
      PrintPaddedLine(2 + row, text, 0);
    }

    DrawHint("arrow keys scroll   g bottom   c cancel job   q exit viewer");
    refresh();
  }

  bool HandleKey(int ch) {
    if (ch == KEY_UP || ch == 'k' || ch == 'K') {
      auto_scroll_ = false;
      scroll_offset_ = std::min(MaxScrollOffset(), scroll_offset_ + 1);
      return false;
    }
    if (ch == KEY_DOWN || ch == 'j' || ch == 'J') {
      scroll_offset_ = std::max(0, scroll_offset_ - 1);
      return false;
    }
    if (ch == KEY_PPAGE) {
      auto_scroll_ = false;
      scroll_offset_ = std::min(MaxScrollOffset(), scroll_offset_ + std::max(1, (LINES - 3) / 2));
      return false;
    }
    if (ch == KEY_NPAGE) {
      scroll_offset_ = std::max(0, scroll_offset_ - std::max(1, (LINES - 3) / 2));
      return false;
    }
    if (ch == 'g' || ch == 'G' || ch == KEY_END) {
      auto_scroll_ = true;
      scroll_offset_ = 0;
      return false;
    }
    if (ch == 'c' || ch == 'C') {
      ShowCancelConfirmation();
      return false;
    }
    return ch == 'q' || ch == 'Q';
  }

  void ShowCancelConfirmation() {
    const int panel_width = std::min(COLS - 8, 44);
    const int panel_height = 4;
    const int panel_top = std::max(4, (LINES - panel_height) / 2);
    const int panel_left = std::max(2, (COLS - panel_width) / 2);

    Draw();
    DrawBox(panel_top, panel_left, panel_height, panel_width, kErrorColorPair);
    PrintClipped(panel_top + 1, panel_left + 2, panel_width - 4, "cancel job? [c=confirm   any other=no]", COLOR_PAIR(kErrorColorPair) | A_BOLD);
    refresh();

    cbreak();
    const int confirm = getch();
    halfdelay(1);
    if (confirm != 'c' && confirm != 'C') {
      return;
    }

    const absl::Status status = client_.CancelJob(job_id_);
    if (!status.ok()) {
      FlashMessage("cancel failed: " + std::string(status.message()), kErrorColorPair);
      return;
    }
    FlashMessage("cancel requested", kOkColorPair);
  }

  void StopAndJoinThreads() {
    stop_.store(true);
    client_.CancelActiveLogStream();
    if (log_thread_.joinable()) {
      log_thread_.join();
    }
    if (status_thread_.joinable()) {
      status_thread_.join();
    }
  }

  std::string job_id_;
  gpu::cli::GpuRunClient& client_;
  std::chrono::steady_clock::time_point started_at_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> done_{false};
  mutable std::mutex ring_mutex_;
  std::deque<std::string> ring_buffer_;
  std::string pending_line_;
  std::mutex status_mutex_;
  std::string state_str_ = "QUEUED";
  std::string status_str_ = "GPUs: none";
  std::thread log_thread_;
  std::thread status_thread_;
  bool auto_scroll_ = true;
  int scroll_offset_ = 0;
};

}  // namespace
bool ScreenServerConfig(JobConfig& config) {
  const std::vector<std::string> address_items = {
      "localhost:50051",
      "192.168.1.10:50051",
      "Enter custom address...",
  };
  const std::vector<std::string> token_items = {
      "No token",
      "Enter token...",
  };

  int address_selection = 0;
  if (config.server_addr == "192.168.1.10:50051") {
    address_selection = 1;
  } else if (!config.server_addr.empty() && config.server_addr != "localhost:50051" &&
             config.server_addr != "127.0.0.1:50051") {
    address_selection = 2;
  }
  int token_selection = config.token.empty() ? 0 : 1;
  int active_section = 0;

  while (!g_interrupted.load()) {
    erase();
    refresh();
    DrawHeader("gpu-run interactive");
    PrintClipped(2, 2, COLS - 4, "Server address", COLOR_PAIR(kHeaderColorPair));
    for (int index = 0; index < static_cast<int>(address_items.size()); ++index) {
      const int attributes = active_section == 0 && index == address_selection
          ? COLOR_PAIR(kOkColorPair) | A_REVERSE
          : COLOR_PAIR(kValueColorPair);
      PrintClipped(3 + index, 4, COLS - 8, address_items[static_cast<std::size_t>(index)], attributes);
    }
    if (!config.server_addr.empty()) {
      PrintClipped(6, 6, COLS - 12, "current: " + config.server_addr, COLOR_PAIR(kHintColorPair) | A_DIM);
    }

    PrintClipped(8, 2, COLS - 4, "Auth token", COLOR_PAIR(kHeaderColorPair));
    for (int index = 0; index < static_cast<int>(token_items.size()); ++index) {
      const int attributes = active_section == 1 && index == token_selection
          ? COLOR_PAIR(kOkColorPair) | A_REVERSE
          : COLOR_PAIR(kValueColorPair);
      PrintClipped(9 + index, 4, COLS - 8, token_items[static_cast<std::size_t>(index)], attributes);
    }
    PrintClipped(12, 6, COLS - 12, config.token.empty() ? "current: <none>" : "current: <hidden>", COLOR_PAIR(kHintColorPair) | A_DIM);
    DrawHint("arrow keys navigate   enter select   q quit wizard");
    refresh();

    const int ch = getch();
    if (IsQuitKey(ch)) {
      return false;
    }
    if (ch == KEY_UP) {
      if (active_section == 0) {
        address_selection = address_selection == 0 ? static_cast<int>(address_items.size()) - 1 : address_selection - 1;
      } else {
        token_selection = token_selection == 0 ? static_cast<int>(token_items.size()) - 1 : token_selection - 1;
      }
      continue;
    }
    if (ch == KEY_DOWN) {
      if (active_section == 0) {
        address_selection = address_selection == static_cast<int>(address_items.size()) - 1 ? 0 : address_selection + 1;
      } else {
        token_selection = token_selection == static_cast<int>(token_items.size()) - 1 ? 0 : token_selection + 1;
      }
      continue;
    }
    if (!IsEnterKey(ch)) {
      continue;
    }

    if (active_section == 0) {
      if (address_selection == 0) {
        config.server_addr = address_items[0];
      } else if (address_selection == 1) {
        config.server_addr = address_items[1];
      } else {
        PrintClipped(14, 2, COLS - 4, "custom address:", COLOR_PAIR(kValueColorPair));
        refresh();
        config.server_addr = PromptTextInput(14, 18, COLS - 20, config.server_addr, true);
        if (config.server_addr.empty()) {
          FlashMessage("server address is required", kErrorColorPair);
          continue;
        }
      }
      active_section = 1;
      continue;
    }

    if (token_selection == 0) {
      config.token.clear();
    } else {
      PrintClipped(14, 2, COLS - 4, "token:", COLOR_PAIR(kValueColorPair));
      refresh();
      config.token = PromptTextInput(14, 10, COLS - 12, config.token, false);
    }
    return true;
  }

  return false;
}

bool ScreenGpuSelect(JobConfig& config, const std::vector<GpuInfo>& gpus) {
  return ShowGpuSelectScreen(config, gpus) == GpuSelectScreenResult::Confirm;
}

bool ScreenJobConfig(JobConfig& config) {
  int active_field = 0;
  while (!g_interrupted.load()) {
    erase();
    refresh();
    DrawHeader("gpu-run job config");

    const std::vector<std::string> labels = {
        "script path",
        "docker image",
        "entrypoint",
        "task type",
        "priority",
    };
    const std::vector<std::string> values = {
        config.script_path.string(),
        config.docker_image,
        config.entrypoint.empty() ? "<image default>" : config.entrypoint,
        FormatTaskType(config.task_type),
        FormatPriority(config.priority),
    };
    const std::vector<std::string> hints = {
        "local file or directory - uploaded to server",
        "must be in config/images.allowlist",
        "blank = image default",
        "left/right to change",
        "left/right to change",
    };

    for (int index = 0; index < static_cast<int>(labels.size()); ++index) {
      const int row = 3 + (index * 2);
      const bool active = index == active_field;
      std::ostringstream label_stream;
      label_stream << std::left << std::setw(16) << labels[static_cast<std::size_t>(index)];
      PrintClipped(row, 2, 16, label_stream.str(), active ? COLOR_PAIR(kOkColorPair) | A_REVERSE : COLOR_PAIR(kHeaderColorPair));
      PrintClipped(row, 20, COLS - 22, values[static_cast<std::size_t>(index)], COLOR_PAIR(kValueColorPair) | (active ? A_REVERSE : 0));
      PrintClipped(row + 1, 20, COLS - 22, hints[static_cast<std::size_t>(index)], COLOR_PAIR(kHintColorPair) | A_DIM);
    }

    DrawHint("arrow keys navigate   enter edit field   left/right cycle   s submit   q quit wizard");
    refresh();

    const int ch = getch();
    if (IsQuitKey(ch)) {
      return false;
    }
    if (ch == KEY_UP) {
      active_field = active_field == 0 ? static_cast<int>(labels.size()) - 1 : active_field - 1;
      continue;
    }
    if (ch == KEY_DOWN) {
      active_field = active_field == static_cast<int>(labels.size()) - 1 ? 0 : active_field + 1;
      continue;
    }
    if ((ch == KEY_LEFT || ch == KEY_RIGHT) && (active_field == 3 || active_field == 4)) {
      if (active_field == 3) {
        config.task_type = config.task_type == gpu::TaskType::COMPUTE ? gpu::TaskType::TRAINING : gpu::TaskType::COMPUTE;
      } else if (ch == KEY_LEFT) {
        if (config.priority == gpu::Priority::LOW) {
          config.priority = gpu::Priority::HIGH;
        } else if (config.priority == gpu::Priority::MEDIUM) {
          config.priority = gpu::Priority::LOW;
        } else {
          config.priority = gpu::Priority::MEDIUM;
        }
      } else {
        if (config.priority == gpu::Priority::LOW) {
          config.priority = gpu::Priority::MEDIUM;
        } else if (config.priority == gpu::Priority::MEDIUM) {
          config.priority = gpu::Priority::HIGH;
        } else {
          config.priority = gpu::Priority::LOW;
        }
      }
      continue;
    }
    if (ch == 's' || ch == 'S') {
      if (config.script_path.empty()) {
        FlashMessage("script path is required", kErrorColorPair);
        continue;
      }

      while (!g_interrupted.load()) {
        erase();
        refresh();
        DrawHeader("gpu-run summary");
        const int box_top = 2;
        const int box_left = 2;
        const int box_width = COLS - 4;
        const int box_height = 12;
        DrawBox(box_top, box_left, box_height, box_width, kValueColorPair);
        const std::vector<std::string> summary_lines = {
            BuildJobConfigSummaryLine("server", config.server_addr),
            BuildJobConfigSummaryLine("script", config.script_path.string()),
            BuildJobConfigSummaryLine("image", config.docker_image),
            BuildJobConfigSummaryLine("entrypoint", config.entrypoint.empty() ? "<image default>" : config.entrypoint),
            BuildJobConfigSummaryLine("task", FormatTaskType(config.task_type)),
            BuildJobConfigSummaryLine("priority", FormatPriority(config.priority)),
            BuildJobConfigSummaryLine("GPU count", std::to_string(config.gpu_count)),
            BuildJobConfigSummaryLine("preferred", FormatGpuIdList(config.preferred_gpu_ids)),
        };
        for (std::size_t index = 0; index < summary_lines.size(); ++index) {
          PrintClipped(box_top + 1 + static_cast<int>(index), box_left + 2, box_width - 4, summary_lines[index], COLOR_PAIR(kValueColorPair));
        }
        PrintClipped(box_top + box_height + 1, 2, COLS - 4, "confirm? [enter=yes   q=cancel]", COLOR_PAIR(kOkColorPair) | A_BOLD);
        DrawHint("q quit wizard");
        refresh();

        const int confirm = getch();
        if (IsEnterKey(confirm)) {
          return true;
        }
        if (IsQuitKey(confirm)) {
          return false;
        }
      }
      return false;
    }
    if (!IsEnterKey(ch)) {
      continue;
    }

    if (active_field == 0) {
      PrintClipped(15, 2, COLS - 4, "script path:", COLOR_PAIR(kValueColorPair));
      refresh();
      config.script_path = PromptTextInput(15, 15, COLS - 17, config.script_path.string(), false);
    } else if (active_field == 1) {
      PrintClipped(15, 2, COLS - 4, "docker image:", COLOR_PAIR(kValueColorPair));
      refresh();
      config.docker_image = PromptTextInput(15, 16, COLS - 18, config.docker_image, false);
    } else if (active_field == 2) {
      PrintClipped(15, 2, COLS - 4, "entrypoint:", COLOR_PAIR(kValueColorPair));
      refresh();
      config.entrypoint = PromptTextInput(15, 15, COLS - 17, config.entrypoint, false);
    }
  }

  return false;
}

void ScreenLogViewer(const std::string& job_id, gpu::cli::GpuRunClient& client) {
  LogViewerSession session(job_id, client);
  session.Run();
}

SessionResult RunWizard(gpu::cli::GpuRunClient& client) {
  SessionResult result;
  const gpu::cli::ClientOptions options = client.GetOptions();

  JobConfig config;
  config.server_addr = options.server_address == "127.0.0.1:50051" ? "localhost:50051" : options.server_address;
  config.token = options.bearer_token.value_or("");

  if (!ScreenServerConfig(config)) {
    return result;
  }

  client.ResetConnection(ToClientOptions(config));

  while (!g_interrupted.load()) {
    auto gpus = client.ListGpus();
    while (!gpus.ok()) {
      if (!ShowRetryQuitPanel(std::string(gpus.status().message()))) {
        return result;
      }
      gpus = client.ListGpus();
    }

    const GpuSelectScreenResult gpu_screen = ShowGpuSelectScreen(config, ToGpuInfos(*gpus));
    if (gpu_screen == GpuSelectScreenResult::Quit) {
      return result;
    }
    if (gpu_screen == GpuSelectScreenResult::Refresh) {
      continue;
    }
    break;
  }

  if (!ScreenJobConfig(config)) {
    return result;
  }

  std::string bundle_id;
  while (!g_interrupted.load()) {
    DrawUploadProgress(0);
    auto uploaded = client.UploadBundleWithProgress(config.script_path, [](std::int64_t bytes_sent) {
      DrawUploadProgress(bytes_sent);
    });
    if (uploaded.ok()) {
      bundle_id = *uploaded;
      break;
    }
    if (!ShowRetryQuitPanel(std::string(uploaded.status().message()))) {
      return result;
    }
  }

  std::string job_id;
  while (!g_interrupted.load()) {
    erase();
    refresh();
    DrawHeader("gpu-run submit");
    PrintClipped(3, 2, COLS - 4, "submitting job...", COLOR_PAIR(kValueColorPair) | A_BOLD);
    DrawHint("submitting job to server");
    refresh();

    auto submitted = client.SubmitJobExplicit(bundle_id, config);
    if (submitted.ok()) {
      job_id = *submitted;
      break;
    }
    if (!ShowRetryQuitPanel(std::string(submitted.status().message()))) {
      return result;
    }
  }

  g_task_type_label = FormatTaskType(config.task_type);
  result.submitted = true;
  result.job_id = job_id;
  result.post_action = ShowPostSubmitActionMenu(job_id);
  result.task_type_label = g_task_type_label;
  return result;
}

int Run(int argc, char** argv) {
  std::signal(SIGINT, SigIntHandler);

  gpu::cli::ClientOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--interactive") {
      continue;
    }
    if (arg == "--server") {
      if (index + 1 >= argc) {
        std::fprintf(stderr, "gpu-run: --server requires a value\n");
        return 2;
      }
      options.server_address = argv[++index];
      continue;
    }
    if (arg == "--token") {
      if (index + 1 >= argc) {
        std::fprintf(stderr, "gpu-run: --token requires a value\n");
        return 2;
      }
      options.bearer_token = std::string(argv[++index]);
      continue;
    }

    std::fprintf(stderr, "gpu-run: interactive mode only accepts --interactive, --server, and --token\n");
    return 2;
  }

  initscr();
  if (LINES < kMinLines || COLS < kMinCols) {
    endwin();
    std::fprintf(stderr, "terminal too small â€” minimum 60x20 required\n");
    return 1;
  }

  start_color();
  use_default_colors();
  init_pair(kHeaderColorPair, COLOR_CYAN, -1);
  init_pair(kOkColorPair, COLOR_GREEN, -1);
  init_pair(kValueColorPair, COLOR_YELLOW, -1);
  init_pair(kErrorColorPair, COLOR_RED, -1);
  init_pair(kStatusBarColorPair, COLOR_WHITE, COLOR_BLACK);
  init_pair(kHintColorPair, COLOR_BLUE, -1);

  cbreak();
  noecho();
  keypad(stdscr, TRUE);
  curs_set(0);

  gpu::cli::GpuRunClient client(options);

  try {
    const SessionResult result = RunWizard(client);
    if (!result.submitted) {
      endwin();
      return 0;
    }

    if (result.post_action == PostSubmitAction::StreamLogs) {
      ScreenLogViewer(result.job_id, client);
      endwin();
      return 0;
    }

    endwin();
    if (result.post_action == PostSubmitAction::PollStatus) {
      while (true) {
        auto status = client.GetStatus(result.job_id);
        if (!status.ok()) {
          std::fprintf(stderr, "%s\n", std::string(status.status().message()).c_str());
          return 1;
        }
        PrintStatusJson(*status);
        if (IsTerminalState(status->state)) {
          return 0;
        }
        std::this_thread::sleep_for(kStatusPollInterval);
      }
    }

    std::printf("%s\n", result.job_id.c_str());
    return 0;
  } catch (...) {
    endwin();
    throw;
  }
}

}  // namespace gpu_run::tui