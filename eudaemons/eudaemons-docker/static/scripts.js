var source = new EventSource("/stream");
source.onmessage = function(event) {
  var terminal = document.getElementById("terminal");
  terminal.innerHTML += event.data + "<br/>";
  terminal.scrollTop = terminal.scrollHeight;
};
source.onerror = function(event) {
  console.error("SSE error", event);
};
