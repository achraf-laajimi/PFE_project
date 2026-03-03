// app.js  -- ClassQuiz frontend logic (ASCII-only source file)
// All Arabic strings use \uXXXX escapes so this file is encoding-safe.

const chatMessages = document.getElementById('chatMessages');
const messageInput  = document.getElementById('messageInput');

// -- Session config --------------------------------------------------
var STUDENT_ID   = "239645";
var SESSION_ID_KEY = "classquiz_session_id";
var HISTORY_BASE   = "classquiz_chat_history";

var currentSessionId = (function () {
    var existing = localStorage.getItem(SESSION_ID_KEY);
    if (existing) return existing;
    var fresh = "session_" + Date.now();
    localStorage.setItem(SESSION_ID_KEY, fresh);
    return fresh;
})();

// -- LocalStorage helpers --------------------------------------------
function saveHistory(messages) {
    try { localStorage.setItem(HISTORY_BASE + "_" + currentSessionId, JSON.stringify(messages)); }
    catch (e) { console.warn("history save failed", e); }
}
function loadHistory() {
    try {
        var raw = localStorage.getItem(HISTORY_BASE + "_" + currentSessionId);
        if (!raw) return [];
        var parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed : [];
    } catch (e) { return []; }
}
var chatHistory = loadHistory();

// -- Markdown + escape helpers ---------------------------------------
function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
        marked.setOptions({ breaks: true, gfm: true });
        return marked.parse(text);
    }
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/- (.+)/g, '<li>$1</li>')
        .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
        .replace(/\n/g, '<br>');
}
function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// -- Session management ----------------------------------------------
async function loadSessions() {
    var list = document.getElementById('sessionList');
    if (!list) return;
    try {
        var res = await fetch('http://localhost:5000/sessions?student_id=' + STUDENT_ID);
        if (!res.ok) throw new Error('sessions fetch failed');
        var sessions = await res.json();
        list.innerHTML = '';
        if (sessions.length === 0) {
            list.innerHTML = '<li class="session-empty">\u0644\u0627 \u062a\u0648\u062c\u062f \u0645\u062d\u0627\u062f\u062b\u0627\u062a \u0633\u0627\u0628\u0642\u0629</li>';
            return;
        }
        sessions.forEach(function (s) {
            var li = document.createElement('li');
            li.className = 'session-item' + (s.session_id === currentSessionId ? ' active' : '');
            li.dataset.sessionId = s.session_id;

            var label = document.createElement('span');
            label.className = 'session-title';
            label.textContent = s.title || '\u0645\u062d\u0627\u062f\u062b\u0629 \u062c\u062f\u064a\u062f\u0629';
            label.onclick = function () { switchSession(s.session_id); };

            var del = document.createElement('button');
            del.className = 'session-delete';
            del.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg>';
            del.title = '\u062d\u0630\u0641';
            del.onclick = function (e) { e.stopPropagation(); deleteSession(s.session_id); };

            li.append(label, del);
            list.appendChild(li);
        });
    } catch (e) { console.warn('Could not load sessions:', e); }
}

function switchSession(sessionId) {
    if (sessionId === currentSessionId) return;
    currentSessionId = sessionId;
    localStorage.setItem(SESSION_ID_KEY, sessionId);
    chatMessages.innerHTML = '';
    chatHistory = loadHistory();
    if (chatHistory.length) {
        chatHistory.forEach(function (msg) { _renderBubble(msg.text, msg.sender); });
    } else {
        _renderWelcome();
    }
    chatMessages.scrollTop = chatMessages.scrollHeight;
    document.querySelectorAll('.session-item').forEach(function (li) {
        li.classList.toggle('active', li.dataset.sessionId === sessionId);
    });
}

function newSession() {
    currentSessionId = "session_" + Date.now();
    localStorage.setItem(SESSION_ID_KEY, currentSessionId);
    chatHistory = [];
    saveHistory(chatHistory);
    chatMessages.innerHTML = '';
    _renderWelcome();
    loadSessions();
}

async function deleteSession(sessionId) {
    try {
        await fetch('http://localhost:5000/sessions/' + sessionId, { method: 'DELETE' });
        localStorage.removeItem(HISTORY_BASE + "_" + sessionId);
        if (sessionId === currentSessionId) { newSession(); } else { loadSessions(); }
    } catch (e) { console.warn('delete session failed', e); }
}

function toggleSidebar() {
    var sidebar = document.getElementById('sidebar');
    if (sidebar) sidebar.classList.toggle('open');
}

// -- Message rendering -----------------------------------------------
function _renderBubble(text, sender) {
    var messageDiv = document.createElement('div');
    messageDiv.className = 'message ' + sender;
    var avatar = sender === 'bot' ? '\uD83E\uDD16' : '\uD83D\uDC64';
    var time = new Date().toLocaleTimeString('ar-EG', { hour: '2-digit', minute: '2-digit' });
    var rendered = sender === 'bot' ? renderMarkdown(text) : escapeHtml(text);
    messageDiv.innerHTML =
        '<div class="message-avatar">' + avatar + '</div>' +
        '<div class="message-content">' +
          '<div class="message-bubble">' + rendered + '</div>' +
          '<div class="message-time">' + time + '</div>' +
        '</div>';
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addMessage(text, sender) {
    _renderBubble(text, sender);
    var time = new Date().toLocaleTimeString('ar-EG', { hour: '2-digit', minute: '2-digit' });
    chatHistory.push({ text: text, sender: sender, time: time });
    saveHistory(chatHistory);
    if (sender === 'user' && chatHistory.filter(function (m) { return m.sender === 'user'; }).length === 1) {
        setTimeout(loadSessions, 800);
    }
}

function _renderWelcome() {
    var t = new Date().toLocaleTimeString('ar-EG', { hour: '2-digit', minute: '2-digit' });
    chatMessages.innerHTML =
        '<div class="message bot">' +
          '<div class="message-avatar">\uD83E\uDD16</div>' +
          '<div class="message-content">' +
            '<div class="message-bubble">' +
              '\u0645\u0631\u062d\u0628\u0627\u064b! \uD83D\uDC4B \u0623\u0646\u0627 \u0645\u0633\u0627\u0639\u062f\u0643 \u0627\u0644\u0630\u0643\u064a \u0641\u064a ClassQuiz. \u0643\u064a\u0641 \u064a\u0645\u0643\u0646\u0646\u064a \u0645\u0633\u0627\u0639\u062f\u062a\u0643 \u0627\u0644\u064a\u0648\u0645\u061f' +
            '</div>' +
            '<div class="message-time">' + t + '</div>' +
          '</div>' +
        '</div>' +
        '<div class="message bot">' +
          '<div class="message-avatar">\uD83E\uDD16</div>' +
          '<div class="message-content">' +
            '<div class="message-bubble">' +
              '\u064a\u0645\u0643\u0646\u0646\u064a \u0645\u0633\u0627\u0639\u062f\u062a\u0643 \u0641\u064a:' +
              '<div class="quick-replies">' +
                '<button class="quick-reply-btn" onclick="sendQuickReply(\'\u0627\u0645\u062a\u062d\u0627\u0646\u0627\u062a\')">\uD83D\uDCDD \u0627\u0644\u0627\u0645\u062a\u062d\u0627\u0646\u0627\u062a</button>' +
                '<button class="quick-reply-btn" onclick="sendQuickReply(\'\u0627\u0644\u0645\u0648\u0627\u062f \u0627\u0644\u062f\u0631\u0627\u0633\u064a\u0629\')">\uD83D\uDCDA \u0627\u0644\u0645\u0648\u0627\u062f \u0627\u0644\u062f\u0631\u0627\u0633\u064a\u0629</button>' +
                '<button class="quick-reply-btn" onclick="sendQuickReply(\'\u0646\u0635\u0627\u0626\u062d \u0627\u0644\u062f\u0631\u0627\u0633\u0629\')">\uD83D\uDCA1 \u0646\u0635\u0627\u0626\u062d \u0627\u0644\u062f\u0631\u0627\u0633\u0629</button>' +
                '<button class="quick-reply-btn" onclick="sendQuickReply(\'\u0627\u0644\u062f\u0639\u0645 \u0627\u0644\u0641\u0646\u064a\')">\uD83D\uDEE0\uFE0F \u0627\u0644\u062f\u0639\u0645 \u0627\u0644\u0641\u0646\u064a</button>' +
              '</div>' +
            '</div>' +
            '<div class="message-time">' + t + '</div>' +
          '</div>' +
        '</div>';
}

// -- API call --------------------------------------------------------
async function sendMessage() {
    var message = messageInput.value.trim();
    if (!message) return;
    addMessage(message, 'user');
    messageInput.value = '';
    showTypingIndicator();
    try {
        var res = await fetch('http://localhost:5000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message, student_id: STUDENT_ID, session_id: currentSessionId })
        });
        if (!res.ok) throw new Error('bad response');
        var data = await res.json();
        hideTypingIndicator();
        addMessage(data.response, 'bot');
        setTimeout(loadSessions, 500);
    } catch (err) {
        hideTypingIndicator();
        addMessage('\u0639\u0630\u0631\u0627\u064b\u060c \u062d\u062f\u062b \u062e\u0637\u0623 \u0641\u064a \u0627\u0644\u0627\u062a\u0635\u0627\u0644 \u0628\u0627\u0644\u062e\u0627\u062f\u0645 \uD83D\uDE14', 'bot');
        console.error('sendMessage error:', err);
    }
}

// -- UI helpers ------------------------------------------------------
function showTypingIndicator() {
    var d = document.createElement('div');
    d.className = 'message bot';
    d.id = 'typingIndicator';
    d.innerHTML =
        '<div class="message-avatar">\uD83E\uDD16</div>' +
        '<div class="message-content"><div class="message-bubble">' +
          '<div class="typing-indicator">' +
            '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>' +
          '</div>' +
        '</div></div>';
    chatMessages.appendChild(d);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
function hideTypingIndicator() {
    var ind = document.getElementById('typingIndicator');
    if (ind) ind.remove();
}
function sendQuickReply(text) { messageInput.value = text; sendMessage(); }
function handleKeyPress(e) { if (e.key === 'Enter') sendMessage(); }
function insertEmoji() {
    var emojis = ['\uD83D\uDE0A', '\u2764\uFE0F', '\uD83D\uDC4D', '\uD83C\uDF89', '\uD83D\uDCDA', '\u2728', '\uD83C\uDF1F', '\uD83D\uDCA1'];
    messageInput.value += emojis[Math.floor(Math.random() * emojis.length)];
    messageInput.focus();
}

// -- Boot ------------------------------------------------------------
window.addEventListener('load', function () {
    if (chatHistory.length) {
        chatMessages.innerHTML = '';
        chatHistory.forEach(function (msg) { _renderBubble(msg.text, msg.sender); });
    }
    chatMessages.scrollTop = chatMessages.scrollHeight;
    loadSessions();
});