// Golden vectors for prompt post-processing — run from the SillyTavern repo
// root:
//   node src-rs/tests/golden/generate_prompt.js
//
// Uses the REAL upstream `src/prompt-converters.js` so the Rust port in
// src-rs/src/prompt.rs can be verified against Node behavior, including its
// quirks (double prefixes on named messages, tool-squash rules, ...).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { setConfigFilePath } from '../../../src/util.js';

setConfigFilePath('./config.yaml');

const { postProcessPrompt, convertTextCompletionPrompt, embedOpenRouterMedia, addOpenRouterSignatures } =
    await import('../../../src/prompt-converters.js');

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const names = {
    charName: 'Seraphina',
    userName: 'User',
    groupNames: ['Alice', 'Bob'],
    startsWithGroupName(message) {
        return this.groupNames.some((name) => message.startsWith(`${name}: `));
    },
};

const clone = (v) => JSON.parse(JSON.stringify(v));

const cases = [
    {
        name: 'plain system/user/assistant',
        messages: [
            { role: 'system', content: 'sys' },
            { role: 'user', content: 'hi' },
            { role: 'assistant', content: 'hello' },
        ],
    },
    {
        name: 'consecutive same role squashing',
        messages: [
            { role: 'system', content: 'sys' },
            { role: 'user', content: 'a' },
            { role: 'user', content: 'b' },
            { role: 'assistant', content: 'c' },
        ],
    },
    {
        name: 'named messages get prefixed twice',
        messages: [
            { role: 'assistant', name: 'Seraphina', content: 'hi' },
            { role: 'user', name: 'User', content: 'hey' },
        ],
    },
    {
        name: 'example_assistant / example_user prefixing',
        messages: [
            { role: 'system', name: 'example_assistant', content: 'yo' },
            { role: 'system', name: 'example_user', content: 'me' },
            { role: 'user', content: 'q' },
        ],
    },
    {
        name: 'group name already present is not duplicated',
        messages: [
            { role: 'system', name: 'example_assistant', content: 'Alice: already' },
            { role: 'user', content: 'q' },
        ],
    },
    {
        name: 'media parts survive as arrays',
        messages: [
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'look' },
                    { type: 'image_url', image_url: { url: 'data:image/png;base64,AAA' } },
                ],
            },
        ],
    },
    {
        name: 'multiple media parts and text interleaved',
        messages: [
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'a' },
                    { type: 'image_url', image_url: { url: 'data:image/png;base64,AAA' } },
                    { type: 'text', text: 'b' },
                    { type: 'image_url', image_url: { url: 'data:image/png;base64,BBB' } },
                ],
            },
        ],
    },
    {
        name: 'empty content becomes empty string and squashes',
        messages: [
            { role: 'user', content: '' },
            { role: 'user', content: 'x' },
        ],
    },
    {
        name: 'empty messages gets placeholder',
        messages: [],
    },
    {
        name: 'missing content key',
        messages: [{ role: 'user' }, { role: 'assistant', content: 'x' }],
    },
    {
        name: 'tool messages demoted when tools disabled',
        messages: [
            { role: 'user', content: 'q' },
            { role: 'tool', tool_call_id: 't1', content: 'result' },
            { role: 'assistant', content: 'a' },
        ],
    },
    {
        name: 'tool messages kept when tools enabled',
        messages: [
            { role: 'user', content: 'q' },
            { role: 'tool', tool_call_id: 't1', content: 'result' },
            { role: 'assistant', content: 'a' },
        ],
    },
    {
        name: 'strict: mid-prompt system demoted',
        messages: [
            { role: 'user', content: 'q' },
            { role: 'system', content: 'mid' },
            { role: 'assistant', content: 'a' },
        ],
    },
    {
        name: 'strict: system-only prompt gets placeholder',
        messages: [{ role: 'system', content: 'sys' }],
    },
    {
        name: 'strict: system followed by assistant gets placeholder',
        messages: [
            { role: 'system', content: 'sys' },
            { role: 'assistant', content: 'a' },
        ],
    },
    {
        name: 'strict: leading assistant gets placeholder',
        messages: [
            { role: 'assistant', content: 'a' },
            { role: 'user', content: 'u' },
        ],
    },
    {
        name: 'single: everything becomes one user message',
        messages: [
            { role: 'system', content: 'sys' },
            { role: 'assistant', content: 'yo' },
            { role: 'user', content: 'hi' },
        ],
    },
    {
        name: 'single: named assistant',
        messages: [{ role: 'assistant', name: 'Seraphina', content: 'yo' }],
    },
    {
        name: 'single: group name already present',
        messages: [{ role: 'assistant', content: 'Alice: yo' }],
    },
    {
        name: 'tool_calls stripped when tools disabled',
        messages: [
            {
                role: 'assistant',
                content: 'a',
                tool_calls: [{ id: 'c1', type: 'function', function: { name: 'f', arguments: '{}' } }],
            },
            { role: 'user', content: 'u' },
        ],
    },
];

const types = ['merge', 'merge_tools', 'semi', 'semi_tools', 'strict', 'strict_tools', 'single'];

const out = { postProcessPrompt: [] };
for (const c of cases) {
    for (const type of types) {
        const input = clone(c.messages);
        let result;
        try {
            result = postProcessPrompt(input, type, names);
        } catch (e) {
            result = { __error: String(e) };
        }
        out.postProcessPrompt.push({
            case: c.name,
            type,
            input: c.messages,
            output: result,
        });
    }
}

// Text completion prompt conversion
out.convertTextCompletionPrompt = [
    {
        input: [
            { role: 'system', content: 'sys' },
            { role: 'system', name: 'narrator', content: 'n' },
            { role: 'user', content: 'hi' },
        ],
        output: convertTextCompletionPrompt([
            { role: 'system', content: 'sys' },
            { role: 'system', name: 'narrator', content: 'n' },
            { role: 'user', content: 'hi' },
        ]),
    },
    { input: 'raw text', output: convertTextCompletionPrompt('raw text') },
    {
        input: [{ role: 'user', content: [{ type: 'text', text: 'x' }] }],
        output: convertTextCompletionPrompt([{ role: 'user', content: [{ type: 'text', text: 'x' }] }]),
    },
];

// embedOpenRouterMedia
const mediaCases = [
    {
        messages: [
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'a' },
                    { type: 'audio_url', audio_url: { url: 'data:audio/wav;base64,QUJD' } },
                ],
            },
        ],
        opts: { audio: true, video: true },
    },
    {
        messages: [
            {
                role: 'user',
                content: [{ type: 'audio_url', audio_url: { url: 'data:audio/mpeg;base64,QUJD' } }],
            },
        ],
        opts: { audio: true, video: false },
    },
    {
        messages: [
            {
                role: 'user',
                content: [{ type: 'audio_url', audio_url: { url: 'https://example.com/a.mp3' } }],
            },
        ],
        opts: { audio: true, video: true },
    },
    {
        messages: [
            {
                role: 'user',
                content: [{ type: 'video_url', video_url: { url: 'data:video/mp4;base64,QUJD' } }],
            },
        ],
        opts: { audio: false, video: true },
    },
];
out.embedOpenRouterMedia = mediaCases.map((c) => {
    const input = clone(c.messages);
    embedOpenRouterMedia(input, c.opts);
    return { input: c.messages, opts: c.opts, output: input };
});

// addOpenRouterSignatures
const sigCases = [
    {
        model: 'openai/gpt-5',
        messages: [
            {
                role: 'assistant',
                content: 'x',
                signature: 'sig1',
                tool_calls: [{ id: 'call_1', type: 'function', signature: 'sig2' }],
            },
        ],
    },
    {
        model: 'anthropic/claude-sonnet-4',
        messages: [{ role: 'assistant', content: 'x', signature: 'sig1' }],
    },
    {
        model: 'some/other-model',
        messages: [{ role: 'assistant', content: 'x', signature: 'sig1' }],
    },
];
out.addOpenRouterSignatures = sigCases.map((c) => {
    const input = clone(c.messages);
    addOpenRouterSignatures(input, c.model);
    return { model: c.model, input: c.messages, output: input };
});

fs.writeFileSync(path.join(__dirname, 'golden_prompt.json'), JSON.stringify(out, null, 2));
console.log('golden_prompt.json written:', path.join(__dirname, 'golden_prompt.json'));
console.log('postProcessPrompt vectors:', out.postProcessPrompt.length);
