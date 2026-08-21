// Terminal chat client. Streams SSE, dims reasoning, keeps multi-turn history.
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "../../third_party/httplib.h"
#include "../../third_party/json.hpp"

using json = nlohmann::json;

int main(int argc, char** argv) {
  std::string host = "127.0.0.1";
  int port = 8090;
  std::string model = "qwen38-27b";
  bool think = true;
  std::string effort;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--host" && i + 1 < argc) host = argv[++i];
    else if (a == "--port" && i + 1 < argc) port = atoi(argv[++i]);
    else if (a == "--model" && i + 1 < argc) model = argv[++i];
    else if (a == "--no-think") think = false;
    else if (a == "--effort" && i + 1 < argc) effort = argv[++i];
    else if (a == "-h" || a == "--help") {
      printf("usage: %s [--host H] [--port N] [--model ID] [--no-think] "
             "[--effort xhigh|medium|low]\n"
             "commands: /new  /think  /nothink  /effort LEVEL  /quit\n", argv[0]);
      return 0;
    }
  }

  httplib::Client cli(host, port);
  cli.set_read_timeout(900, 0);
  json history = json::array();

  printf("\033[1mqwen38-3090-cuda-server\033[0m  %s:%d  model=%s  thinking=%s\n",
         host.c_str(), port, model.c_str(), think ? "on" : "off");
  printf("commands: /new  /think  /nothink  /effort LEVEL  /quit\n\n");

  std::string line;
  while (true) {
    printf("\033[1;32myou>\033[0m ");
    fflush(stdout);
    if (!std::getline(std::cin, line)) break;
    if (line.empty()) continue;
    if (line == "/quit" || line == "/exit") break;
    if (line == "/new") { history = json::array(); printf("(history cleared)\n\n"); continue; }
    if (line == "/think") { think = true; printf("(thinking on)\n\n"); continue; }
    if (line == "/nothink") { think = false; printf("(thinking off)\n\n"); continue; }
    if (line.rfind("/effort ", 0) == 0) { effort = line.substr(8); printf("(effort=%s)\n\n", effort.c_str()); continue; }

    history.push_back({{"role", "user"}, {"content", line}});
    json body{{"model", model}, {"messages", history}, {"stream", true},
              {"max_tokens", 2048}, {"enable_thinking", think}};
    if (!effort.empty()) body["reasoning_effort"] = effort;

    std::string content, reasoning, buf;
    bool in_reason = false;
    printf("\033[1;36mbot>\033[0m ");
    fflush(stdout);
    cli.Post("/v1/chat/completions", httplib::Headers{}, body.dump(), "application/json",
             [&](const char* data, size_t len) {
               buf.append(data, len);
               size_t p;
               while ((p = buf.find("\n\n")) != std::string::npos) {
                 const std::string frame = buf.substr(0, p);
                 buf.erase(0, p + 2);
                 if (frame.rfind("data:", 0) != 0) continue;
                 const std::string payload = frame.substr(5 + (frame[5] == ' ' ? 1 : 0));
                 if (payload == "[DONE]") continue;
                 json j;
                 try { j = json::parse(payload); } catch (...) { continue; }
                 for (auto& c : j.value("choices", json::array())) {
                   const auto d = c.value("delta", json::object());
                   if (d.contains("reasoning_content")) {
                     if (!in_reason) { printf("\033[2m"); in_reason = true; }
                     const std::string s = d["reasoning_content"].get<std::string>();
                     reasoning += s; fputs(s.c_str(), stdout);
                   }
                   if (d.contains("content")) {
                     if (in_reason) { printf("\033[0m"); in_reason = false; }
                     const std::string s = d["content"].get<std::string>();
                     content += s; fputs(s.c_str(), stdout);
                   }
                   if (d.contains("tool_calls"))
                     printf("\n\033[1;33m[tool_calls]\033[0m %s", d["tool_calls"].dump().c_str());
                 }
               }
               fflush(stdout);
               return true;
             });
    if (in_reason) printf("\033[0m");
    printf("\n\n");
    // History carries `content` only; the server re-renders thinking from
    // reasoning_content when preserve_thinking asks for it.
    history.push_back({{"role", "assistant"}, {"content", content}});
  }
  return 0;
}
