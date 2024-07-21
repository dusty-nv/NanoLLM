
var grid;
var drawflow;
var pluginTypes;
var moduleTypes;
var nodeIdToName = {};
var stateListeners = {};
var outputListeners = {};
var ignoreGraphEvents=false;


function addGrid() {
  grid = GridStack.init({'column': 12, 'cellHeight': 50, 'float': true});

  grid.on('added', function(event, items) {
    items.forEach((item) => {
      console.log(`grid widget ${item.id} was added`, item);
      sendWebsocket({
        'config_plugin': {
          'name': item.id,
          'layout_grid': {'x': item.x, 'y': item.y, 'w': item.w, 'h': item.h}
      }});
    });
  });
  
  grid.on('change', function(event, items) {
    items.forEach((item) => {
      console.log(`grid widget ${item.id} changed position/size`, item);
      sendWebsocket({
        'config_plugin': {
          'name': item.id,
          'layout_grid': {'x': item.x, 'y': item.y, 'w': item.w, 'h': item.h}
      }});
    });
  });
  
  grid.on('removed', function(event, items) {
    items.forEach((item) => {
      console.log(`grid widget ${item.id} has been removed`, item);
      sendWebsocket({
        'config_plugin': {
          'name': item.id,
          'layout_grid': {} 
      }});
    });
  });
  
  
  /*grid.on('resize', function(event, el) {
    fitGridWidgetContents(el);
  });*/

  /*addGraphEditor();

  drawflow = new Drawflow(document.getElementById("drawflow"));
  drawflow.start();
  
  drawflow.on('connectionCreated', onNodeConnectionCreated);
  drawflow.on('connectionRemoved', onNodeConnectionRemoved);*/
}

function addGridWidget(id, title, html, titlebar_html, grid_options) {
  const plugin = id.includes('_grid') ? id.replace('_grid', '') : id;
  
  if( grid_options == undefined ) 
      grid_options = {w: 3, h: 3};

  if( titlebar_html != undefined )
      titlebar_html = `<span class="float-top">${titlebar_html}</span>`;
   
  let title_html = '';
  
  if( title != undefined || titlebar_html != undefined ) {
    title_html += `<span class="mb-2">`;
    
    if( title != undefined )
      title_html += `<h5 class="d-inline">${title}</h5>`;

    title_html += `<button id="${id}_close" type="button" class="btn-close float-end float-top ms-2" aria-label="Close"></button>`;
    
    if( document.getElementById(`${title}_config_dialog`) != null )
      title_html += `<i id="${id}_show_config" class="fa fa-cog float-end gear-button" aria-hidden="true"></i>`;
    
    if( titlebar_html != undefined )
      title_html += titlebar_html;

    title_html += `</span>`;
  }    
  //  position: absolute; right: 15px;
  /*let card_html = `
  <div class="card ms-1 mt-1">
    <div class="card-body d-flex flex-column h-100">
      <div class="card-title" data-bs-toggle="collapse" href="${id}" role="button">
        <h5 class="d-inline">${title}</h5>
        <span class="float-end float-top fa fa-chevron-circle-down fa-lg mt-1 ms-1 me-1" data-bs-toggle="collapse" href="#${id}" id="${id}_collapse_btn" title="Minimize"></span>
        ${titlebar_html}
      </div>
      <div class="collapse show d-flex flex-column h-100" id="${id}">
          ${html}
      </div>
    </div>
  </div>
  `;*/

  let card_html = `
  <div class="card" id="${id}">
    <div class="card-body d-inline-flex flex-column h-100">
      ${title_html}
      ${html}
    </div>
  </div>
  `;
  
  if( ! ('id' in grid_options) )
    grid_options['id'] = plugin;
    
  const widget = grid.addWidget(card_html, grid_options);
  
  console.log(`created grid widget id=${id} title=${title}`, widget);
  
  let btn_close = document.getElementById(`${id}_close`);
  
  if( btn_close != undefined ) {
      btn_close.addEventListener('click', function(e) {
      grid.removeWidget(widget);
    });
  }
  
  let btn_config = document.getElementById(`${id}_show_config`);
  
  if( btn_config != undefined ) {
      btn_config.addEventListener('click', function(e) {
        sendWebsocket({'get_state_dict': title});
        const config_modal = new bootstrap.Modal(`#${title}_config_dialog`);
        config_modal.show();
    });
  }
  
  addStateListener(plugin, function(state_dict) {
    if( 'layout_grid' in state_dict ) {
      console.log(`updating ${plugin} grid widget`, state_dict['layout_grid']);
      grid.update(widget, state_dict['layout_grid']);
    }
  });
  
  sendWebsocket({'get_state_dict': plugin});
  return widget;
} 

function addTextInputWidget(name, id, title, grid_options) {
  const input_id = `${id}_input`;
  const submit_id = `${id}_submit`;
  
  const html = `
    <div class="input-group">
      <textarea id="${input_id}" class="form-control" rows="2" placeholder="Enter to send (Shift+Enter for newline)"></textarea>
      <span id="${submit_id}" class="input-group-text bg-light-gray bi bi-arrow-return-left" style="color: #eeeeee;"></span>
    </div>
  `;
  
  let widget = addGridWidget(id, null, html, null, Object.assign({w: 4, h: 2}, grid_options));

  let onsubmit = function() {
    const input = document.getElementById(input_id);
    console.log('submitting text input', input.value);
    msg = {}
    msg[name] = {'input': input.value};
    sendWebsocket(msg);
    input.value = "";
  }
  
  let onkeydown = function(event) {
    // https://stackoverflow.com/a/49389811
    if( event.which === 13 && !event.shiftKey ) {
      if( !event.repeat )
        onsubmit();
      event.preventDefault(); // prevents the addition of a new line in the text field
    }
  }

  document.getElementById(input_id).addEventListener('keydown', onkeydown);
  document.getElementById(submit_id).addEventListener('click', onsubmit);

  return widget;
}

function scrollBottom(container) {  // https://stackoverflow.com/a/21067431
  container.scrollTop = container.scrollHeight - container.clientHeight;
  //console.log(`scrolling to bottom ${container.scrollTop} ${container.scrollHeight} ${container.clientHeight}`);
}

function addTextStreamWidget(name, id, title, grid_options) {
  const history_id = `${id}_history`;

  const html = `
    <div id="${history_id}" class="bg-medium-gray p-2 mb-2" style="font-family: monospace, monospace; font-size: 100%; overflow-y: scroll; flex-grow: 1;"</div>
  `;
  
  let widget = addGridWidget(id, title, html, null, Object.assign({w: 4, h: 6}, grid_options));

  addStateListener(name, function(state_dict) {
    if( ! ('text' in state_dict) )
      return;
    
    let el_type = 'p';  
    
    if( 'delta' in state_dict )
      el_type = 'span';

    let chc = document.getElementById(history_id);
    let isScrolledToBottom = chc.scrollHeight - chc.clientHeight <= chc.scrollTop + 1;
    
    chc.innerHTML += `
      <${el_type} style="color: ${state_dict['color']}">${state_dict['text']}</${el_type}>
    `;
    
    if( isScrolledToBottom ) { // autoscroll unless the user has scrolled up
      scrollBottom(chc);
    }
  });
  
  return widget;
}

function terminalTextColor(level) {
    if( level == 'success' ) return 'limegreen';
    else if( level == 'error' ) return 'red';
    else if( level == 'warning' ) return 'orange';
    else if( level == 'info' ) return 'rgb(225,225,225)'
    else return 'rgb(180,180,180)';
}
        
function addTerminalWidget(name, id, title, grid_options) {
  const history_id = `${id}_history`;

  const html = `
    <div id="${history_id}" class="bg-medium-gray p-2 mb-2" style="font-family: monospace, monospace; font-size: 100%; overflow: scroll; text-wrap: nowrap; flex-grow: 1;"</div>
  `;
  
  let widget = addGridWidget(id, 'Terminal', html, null, Object.assign({x: 0, y: 14, w: 8, h: 6}, grid_options));

  addOutputListener(name, 0, function(log_entry) {
    let chc = document.getElementById(history_id);
    let isScrolledToBottom = chc.scrollHeight - chc.clientHeight <= chc.scrollTop + 1;
    
    const level = log_entry['level'];
    const msg = log_entry['message'];

    const html = `<p class="m-0 p-0" style="color: ${terminalTextColor(level)}">${escapeHTML(msg)}</p>`;
    
    if( chc.innerHTML.length < 10000 )
      chc.innerHTML += html;
    else
      chc.innerHTML = html;

    if( isScrolledToBottom ) { // autoscroll unless the user has scrolled up
      scrollBottom(chc);
    }
  });
  
  return widget;
}

// https://stackoverflow.com/a/6234804
function escapeHTML(unsafe) {
  return unsafe
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
    .replaceAll('\n', '<br/>');
}

function updateNanoDB(gallery_id, search_results) { 
  //console.log(`updateNanoDB(${gallery_id}) => `, search_results, window.location);
  
  let obj = $(`#${gallery_id}`);
  let contents = '';
  
  obj.empty();
  
  for( let n=0; n < search_results.length; n++ ) {
    const result = search_results[n];
    contents += `<div class="me-1" style="position: relative; color: white; display: inline;">`;
    contents += `<img src="${window.location.origin + '/' + result['metadata']['path']}" style="aspect-ratio: 1.47; width: 24%;" title="&quot;METADATA&quot; : ${escapeHTML(JSON.stringify(result['metadata'], null, 2))}">`;
    contents += `</div>`;
      //contents = '<div style="position: relative; text-align: center; color: white" width="50%">';
      //let contents = `<div class="me-1" style="position: relative; color: white; display: inline;">`;  // 49% width, -288% translate for 2x4 grid
      //contents += `<img src=${search_results[n]['image']} style="aspect-ratio: 1.47; width: 32%;" title="&quot;METADATA&quot; : ${search_results[n]['metadata']}">`;
      
      //contents += `<img src=${search_results[n]['image']} style="aspect-ratio: 1.47; width: 32%;" title="&quot;METADATA&quot; : ${search_results[n]['metadata']}">`;
      //contents += `<div class="ps-1 pe-1" style="position: absolute; top: 0%; right: 0%; transform: translate(0%, -168%); background-color: rgba(0,0,0,0.5)">${search_results[n]['similarity']}</div></div>`;
      //contents += '</div>';
    
  }
  
  obj.append(contents);
}


function addNanoDBWidget(name, id, title, grid_options) {
  const gallery_id = `${id}_results`;
  const input_id = `${id}_input`;
  const submit_id = `${id}_submit`;
  
  const html = `
    <div class="input-group">
      <input id="${input_id}" class="form-control" placeholder="Enter a search query"></input>
      <span id="${submit_id}" class="input-group-text bg-light-gray bi bi-arrow-return-left" style="color: #eeeeee;"></span>
    </div>
    <div id="${gallery_id}" class="bg-medium-gray p-2 mt-2" style="font-size: 100%; overflow-y: scroll; flex-grow: 1;" ondrop="onFileDrop(event)" ondragover="onFileDrag(event)"></div>
  `;
  
  let widget = addGridWidget(id, title, html, null, Object.assign({w: 4, h: 10}, grid_options));

  let onsubmit = function() {
    const input = document.getElementById(input_id);
    console.log('submitting to NanoDB:', input.value);
    msg = {};
    msg[name] = {'input': input.value};
    sendWebsocket(msg);
    //input.value = "";
  }
  
  let onkeydown = debounce(function(event) {
    if( !event.repeat )
        onsubmit();
  }, 100)

  document.getElementById(input_id).addEventListener('keydown', onkeydown);
  document.getElementById(submit_id).addEventListener('click', onsubmit);

  addOutputListener(name, 0, function(search_results) {
    updateNanoDB(gallery_id, search_results);
  });

  msg = {};
  msg[name] = {'input': 'a polar bear swimming in the water'};

  sendWebsocket(msg);
  return widget;
}

function addChatWidget(name, id, title, grid_options) {
  const history_id = `${id}_history`;
  const input_id = `${id}_input`;
  const submit_id = `${id}_submit`;
  
  const html = `
    <div id="${history_id}" class="bg-medium-gray p-2 mb-2" style="font-size: 100%; overflow-y: scroll; flex-grow: 1;" ondrop="onFileDrop(event)" ondragover="onFileDrag(event)"></div>
    <div class="input-group">
      <textarea id="${input_id}" class="form-control" rows="1" placeholder="Enter to send (Shift+Enter for newline) &nbsp;&nbsp;&nbsp; [Commands: &nbsp; /reset /refresh]"></textarea>
      <span id="${submit_id}" class="input-group-text bg-light-gray bi bi-arrow-return-left" style="color: #eeeeee;"></span>
    </div>
  `;
  
  let widget = addGridWidget(id, title, html, null, Object.assign({w: 4, h: 14}, grid_options));

  let onsubmit = function() {
    const input = document.getElementById(input_id);
    console.log('submitting chat message:', input.value);
    msg = {};
    msg[name] = {'input': input.value};
    sendWebsocket(msg);
    input.value = "";
  }
  
  let onkeydown = function(event) {
    // https://stackoverflow.com/a/49389811
    if( event.which === 13 && !event.shiftKey ) {
      if( !event.repeat )
        onsubmit();
      event.preventDefault(); // prevents the addition of a new line in the text field
    }
  }

  document.getElementById(input_id).addEventListener('keydown', onkeydown);
  document.getElementById(submit_id).addEventListener('click', onsubmit);

  addOutputListener(name, 4, function(history) {
    updateChatHistory(history_id, history);
  });
  
  msg = {};
  msg[name] = {'input': '/refresh'};
  sendWebsocket(msg);

  return widget;
}

function updateChatHistory(id, history) {    
    let chj = $(`#${id}`);
    let chc = document.getElementById(id);
    let isScrolledToBottom = chc.scrollHeight - chc.clientHeight <= chc.scrollTop + 1;

    chj.empty(); // clear because server may remove partial/rejected ASR prompts
    
    for( let n=0; n < history.length; n++ ) {
      const role = history[n]['role'];
      
      /*if( role == 'system' )
        continue;*/
        
      let contents = '';
      var hasImage = 'image' in history[n];
      
      if( hasImage ) {
        contents += `<img src=${history[n]['image']} width="100%">`;
        imageAtBottom = true;
      }
      
      if( 'text' in history[n] )
        contents += history[n]['text'];

      if( contents.length > 0 )
        chj.append(`<div id="msg_${n}" class="chat-message-${role} mb-3">${contents}</div><br/>`);
    }

    if( isScrolledToBottom ) { // autoscroll unless the user has scrolled up
      if( hasImage )
        setTimeout(scrollBottom, 50, chc);  // wait for images to load to get right height
      else
        scrollBottom(chc);
    }
}

function addVideoOutputWidget(name, id, title, grid_options) {
  const video_id = `${id}_video_player`;
  const html = `
    <div>
      <video id="${video_id}" autoplay controls playsinline muted>Your browser does not support video</video>
    </div>
  `;
  
  let widget = addGridWidget(id, title, html, null, Object.assign({w: 5, h: 5}, grid_options));
  let video = document.getElementById(video_id);
  
  video.addEventListener('playing', function(e) {
    const abs_rect = video.getBoundingClientRect();
    const options = {'w': Math.ceil(abs_rect.width/grid.cellWidth()), 'h': Math.ceil(abs_rect.height/grid.getCellHeight())+1};
    console.log(`${video_id} playing`, e, abs_rect, options, grid.cellWidth(), grid.getCellHeight());
    grid.update(widget, options);
  });

  let streamName = "output";
  let streamIndex = name.search('_');
  
  if( streamIndex >= 0 ) {
    streamName += name.slice(streamIndex);
  }
  
  playStream(getWebsocketURL(streamName), video); // TODO handle actual stream name
  
  return widget;
}

function addOutputListener(name, channel, listener) {
  if( ! (name in outputListeners) ) {
    outputListeners[name] = [listener];
    msg = {};
    msg[name] = {'send_client_output': channel};
    sendWebsocket(msg);
  }
  else
    outputListeners[name].push(listener);
}

function sendOutput(output) {
  const plugin_name = output['name'];
  if( plugin_name in outputListeners )
    outputListeners[plugin_name].forEach((listener) => {listener(output['data'])});
}

function addStateListener(name, listener) {
  if( name == undefined )
    name = 'all';
    
  if( ! (name in stateListeners) )
    stateListeners[name] = [listener];
  else
    stateListeners[name].push(listener);
}

function setStateDict(state_dicts) {
  for( plugin_name in state_dicts ) {
    const state_dict = state_dicts[plugin_name];
    for( state_name in state_dict ) {
      const id = document.getElementById(`${plugin_name}_config_${state_name}`);
      if( id != null ) {
        id.value = state_dict[state_name];
      }
    }
    
    if( plugin_name in stateListeners )
      stateListeners[plugin_name].forEach((listener) => {listener(state_dict);});

    if( 'all' in stateListeners )
      stateListeners['all'].forEach((listener) => {listener(state_dict);});
  }
}

function setStats(stats_dicts) {
  for( plugin_name in stats_dicts ) {
    const stats = stats_dicts[plugin_name];
    
    if( ! ('summary' in stats) )
      continue;
    
    let summary = stats['summary'];
    
    if( Array.isArray(summary) ) {
      const num_outputs = $(`.${plugin_name} .outputs .output`).length;
      summary = summary.join($(`.${plugin_name}`).length == 0 || num_outputs > 0 ? '<br/>' : ' ');
    }
    
    let node_stats = document.getElementById(`${plugin_name}_node_stats`);
    
    if( node_stats != undefined ) 
      node_stats.innerHTML = summary;
  }
}

function truncate(str, max_len) {
  if( str.length > max_len )
    return str.slice(0, max_len+1);
  else
    return str;
}

function addPlugin(plugin) {
  console.log('addPlugin() =>');
  console.log(plugin)
  
  const plugin_name = plugin['name'];
  const plugin_title = plugin['title'];
  const layout_grid = plugin['layout_grid'];
  const layout_node = plugin['layout_node'];
  const nodes = document.querySelectorAll('.drawflow-node');

  if( ('x' in layout_node) && ('y' in layout_node) ) {
    var x = layout_node['x'];
    var y = layout_node['y'];
  }
  else
  {
    var x = 10;
    var y = 10;
      
    if( nodes.length > 0 ) {
      const node = nodes[nodes.length-1];
      const abs_rect = node.getBoundingClientRect();
      x = node.offsetLeft + abs_rect.width + 60;
      y = node.offsetTop
    }
  }
  
  let stats_class = '';
  
  if( plugin['outputs'].length > 1 )
    stats_class = 'mt-2';
    
  const html = `
    <div style="position: absolute; top: 5px;">
      ${truncate(plugin_title, 14)}
      <p id="${plugin_name}_node_stats" class="${stats_class}" style="font-family: monospace, monospace; font-size: 80%"></p>
    </div>
  `;
  
  const node_id = drawflow.addNode(plugin_name, plugin['inputs'].length, plugin['outputs'].length, x, y, plugin_name, {}, html);
  nodeIdToName[node_id] = plugin_name;
  console.log(`added node id=${node_id} for {plugin_name}`);
  
  let style = '';
  let max_output_chars = 0;
  
  for( i in plugin['outputs'] ) {
    const output = plugin['outputs'][i].toString();
    max_output_chars = Math.max(max_output_chars, output.length);
    style += `.${plugin_name} .outputs .output:nth-child(${Number(i)+1}):before {`;
    style += `display: block; content: "${output}"; position: relative; min-width: 160px; font-size: 80%; bottom: 2px; right: ${(output.length-1) * 6 + 20}px;} `;
  }
    
  style += `.outputs { margin-top: 20px; font-family: monospace, monospace; } `;
  
  var style_el = document.createElement('style');
  style_el.id = `${plugin_name}_node_io_styles`;
  style_el.innerHTML = style;
  document.head.appendChild(style_el);

  const has_config_dialog = (Object.keys(plugin['parameters']).length > 0);
  
  $(`.${plugin_name}`).on('dblclick', function () {
    //console.log(`double-click ${plugin_name}`);
    if( addPluginGridWidget(plugin_name, plugin['type'], plugin_title) == null ) {
      if( has_config_dialog ) {
        sendWebsocket({'get_state_dict': plugin_name});
        const config_modal = new bootstrap.Modal(`#${plugin['name']}_config_dialog`);
        config_modal.show();
      }
    }
  });
  
  if( has_config_dialog )
    addPluginDialog(plugin_name, 'config', plugin_title, null, plugin['parameters']);
    
  if( Object.keys(layout_grid).length > 0 ) {
    addPluginGridWidget(plugin_name, plugin['type'], plugin_title, layout_grid);
  }
}

function addPluginGridWidget(name, type, title, grid_options) {
  const id = `${name}_grid`;
  
  if( type == undefined )
    type = name;
  
  if( title == undefined )
    title = name;
      
  if( document.getElementById(id) != null )
    return null;
    
  switch(type) {
    case 'UserPrompt':
      return addTextInputWidget(name, id, title, grid_options);
    case 'TextStream':
      return addTextStreamWidget(name, id, title, grid_options);
    case 'NanoLLM':
      return addChatWidget(name, id, title, grid_options);
    case 'VideoOutput':
      return addVideoOutputWidget(name, id, title, grid_options);
    case 'NanoDB':
      return addNanoDBWidget(name, id, title, grid_options);
    case 'TerminalPlugin':
      return addTerminalWidget(name, id, title, grid_options);
    case 'GraphEditor':
      return addGraphEditor(name, id, title, grid_options);
    default:
      return null;
  }
}

function connectPlugins(connections) {
  if( !Array.isArray(connections) )
    connections = [connections];
    
  ignoreGraphEvents = true;
  
  connections.forEach((connection) => {
    const from_id = drawflow.getNodesFromName(connection['from'])[0];
    const to_id = drawflow.getNodesFromName(connection['to'])[0];
    console.log(`connecting plugin ${connection['from']} (id=${from_id}, channel=${connection['output']}) => ${connection['to']} (id=${to_id}, channel=${connection['input']})`);
    drawflow.addConnection(from_id, to_id, `output_${connection['output']+1}`, `input_${connection['input']+1}`);
  });

  ignoreGraphEvents = false;
}

function removePlugins(plugins) {
  if( !Array.isArray(plugins) )
    plugins = [plugins];
    
  ignoreGraphEvents = true;
  
  plugins.forEach((plugin) => {
    try {
      const id = drawflow.getNodesFromName(plugin)[0];
      console.log(`removing plugin ${plugin} (id=${id})`);
      if( plugin in outputListeners )
        delete outputListeners[plugin];
      drawflow.removeNodeId(`node-${id}`);
    } catch(e) { console.log(e); }
  });

  ignoreGraphEvents = false;
}

function addPlugins(plugins) {
  if( !Array.isArray(plugins) )
    plugins = [plugins];
    
  for( i in plugins ) {
    let plugin = plugins[i];
    if( ('global' in plugin) ) {
      addPluginGridWidget(plugin['name'], null, ('layout_grid' in plugin) ? plugin['layout_grid'] : null);
    }
    else
      addPlugin(plugin);
  }
  
  ignoreGraphEvents = true;
  
  for( i in plugins ) {
    for( l in plugins[i]['connections'] ) {
      const conn = plugins[i]['connections'][l];
      console.log('adding connection', conn);

      drawflow.addConnection(
        drawflow.getNodesFromName(plugins[i]['name'])[0], 
        drawflow.getNodesFromName(conn['to'])[0],  
        `output_${conn['output']+1}`, 
        `input_${conn['input']+1}`
      );
    }
  }
  
  ignoreGraphEvents = false;
}

function addPluginTypes(types) {
  pluginTypes = types;

  /*for( pluginName in pluginTypes ) {
    const plugin = pluginTypes[pluginName];

    if( 'init' in plugin && Object.keys(plugin['init']['parameters']).length > 0 ) {
      addPluginDialog(pluginName, 'init', null, plugin['init']['description'], plugin['init']['parameters']);
    }
  }*/
}

function addModules(modules) {
  moduleTypes = modules;

  for( moduleName in moduleTypes ) {
    for( pluginName in moduleTypes[moduleName] ) {
      const plugin = moduleTypes[moduleName][pluginName];
      console.log('addModules', pluginName, plugin);
      if( 'init' in plugin && Object.keys(plugin['init']['parameters']).length > 0 ) {
        addPluginDialog(pluginName, 'init', null, plugin['init']['description'], plugin['init']['parameters']);
      }
    }
  }
}

function capitalizeTitle(string) {
    if( string.length < 4 )
      return string.toUpperCase();
    else
      return string.charAt(0).toUpperCase() + string.slice(1);
}

function pluginMenuBar() {
  if( moduleTypes == undefined )
    return '';
 
  let html = '';
  
  for( module in moduleTypes ) {
    html += `
      <span>
      <a class="dropdown-toggle ms-3" href="#" data-bs-target="#{add_plugin_${module}_dropdown}" data-bs-toggle="dropdown" aria-expanded="false" style="color: #aaaaaa;">
        ${capitalizeTitle(module)}
      </a>
      <span class="dropdown float-end float-top" id="add_plugin_${module}_dropdown">
      <ul class="dropdown-menu" id="add_plugin_${module}_menu">
    `;
    
    for( plugin in moduleTypes[module] )
      html += `<li><a class="dropdown-item" id="menu_create_plugin_${plugin}" href="#">${plugin}</a></li>`; // data-bs-toggle="modal" data-bs-target="#${plugin}_init_dialog"

    html += `</ul></span></span>`;
  }

  return html; 
}  

// <li class="divider">
function pluginContextMenu() {
  if( moduleTypes == undefined )
    return '';
 
  let html = `
    <ul id="plugin_context_menu" class="nav-item dropdown-menu" role="menu" style="display:none">
      <li><a class="dropdown-item" href="#">Presets</a>
          <ul class="submenu dropdown-menu" id="plugin_context_menu_presets"></ul>
      </li>
      <li><a class="dropdown-item" href="#" onclick="removeAll()">Remove All</a></li>
      <li><hr class="dropdown-divider"></li> 
  `;

  for( module in moduleTypes ) {
    html += `
      <li><a class="dropdown-item" href="#">${capitalizeTitle(module)}</a>
        <ul class="submenu dropdown-menu">
    `;
    
    for( plugin in moduleTypes[module] )
      html += `<li><a class="dropdown-item" id="plugin_context_menu_${plugin}" href="#">${plugin}</a></li>`;

    html += `</ul></li>`;
  }
    
  html += `</ul>`;
  return html; 
}  

function nodeLayoutInsertPos() {
  var contextMenu = document.getElementById('plugin_context_menu');
  var nodeEditor = document.getElementById('drawflow').getBoundingClientRect();
  
  var pos = {
    x: parseInt(contextMenu.style.left) - nodeEditor.left, 
    y: parseInt(contextMenu.style.top) - nodeEditor.top,
  };
  
  if( isNaN(pos.x) || isNaN(pos.y) ) {
    const nodes = document.querySelectorAll('.drawflow-node');

    var x = 10;
    var y = 10;
      
    if( nodes.length > 0 ) {
      const node = nodes[nodes.length-1];
      const abs_rect = node.getBoundingClientRect();
      pos.x = node.offsetLeft + abs_rect.width + 60;
      pos.y = node.offsetTop;
    }
    else
    {
      pos.x = 10;
      pos.y = 10;
    }
  }

  return pos;
}

function addGraphEditor(name, id, grid_options) {

  let titlebar_html = pluginMenuBar();
  
  let html = `
    <div id="drawflow" class="mt-1 flex-grow-1 w-100"></div>
  `;

  let widget = addGridWidget(
     id, "Node Editor", 
     html, titlebar_html, 
     Object.assign({w: 8, h: 11}, grid_options)
  );
  
  let editor = document.getElementById("drawflow");
  
  editor.addEventListener('mouseenter', (event) => {
    //console.log('onmouseleave(drawflow) => disabling grid move/resize');
    grid.disable();
  });
  
  editor.addEventListener('mouseleave', (event) => {
    //console.log('onmouseleave(drawflow) => enabling grid move/resize');
    grid.enable();
  });
  
  drawflow = new Drawflow(editor);
  drawflow.start();
  
  drawflow.on('nodeMoved', onNodeMoved);
  drawflow.on('nodeRemoved', onNodeRemoved);
  
  drawflow.on('connectionCreated', onNodeConnectionCreated);
  drawflow.on('connectionRemoved', onNodeConnectionRemoved);
  
  //drawflow.on('contextmenu', onNodeContextMenu);
  document.body.appendChild(fromHTML(pluginContextMenu()));

  for( pluginName in pluginTypes ) {
    const plugin = pluginTypes[pluginName];
    
    let pluginMenus = [
      document.getElementById(`menu_create_plugin_${pluginName}`),
      document.getElementById(`plugin_context_menu_${pluginName}`),
    ];
    
    pluginMenus.forEach(pluginMenu => {
      if( 'init' in plugin && Object.keys(plugin['init']['parameters']).length > 0 ) {
        pluginMenu.addEventListener('click', (event) => {
          console.log(`opening init dialog for ${plugin['name']}`);
          const modal = new bootstrap.Modal(`#${plugin['name']}_init_dialog`);
          modal.show();
        });
      }
      else {
        pluginMenu.addEventListener('click', (event) => {
          sendWebsocket({
            'add_plugin': {
              'type': plugin['name'],
              'layout_node': nodeLayoutInsertPos(),
            }
          });
        });
      }
    });
  }
  
  $(".drawflow").contextMenu({
    menuSelector: "#plugin_context_menu",
    menuSelected: function (invokedOn, selectedMenu) {
        //var msg = "You selected the menu item '" + selectedMenu.text() + "' on the value '" + invokedOn.text() + "'";
        console.log('selected context menu', selectedMenu, msg);
    }
  });

  return widget;
}

function onNodeMoved(id) {
  const node = drawflow.getNodeFromId(id);
  //console.log('node moved', id, node);
  sendWebsocket({
    'config_plugin': {
      'name': node['name'],
      'layout_node': {'x': node['pos_x'], 'y': node['pos_y']}
    } 
  });
}

function onNodeRemoved(id) {
  const pluginName = nodeIdToName[id];
  delete nodeIdToName[id];
  console.log(`node removed id=${id} name=${pluginName}`);
  
  if( !ignoreGraphEvents )
    sendWebsocket({'remove_plugin': pluginName});
}

function onNodeConnectionCreated(event) {
  console.log(`onNodeConnectionCreated(ignore=${ignoreGraphEvents})`, event);
  
  if( ignoreGraphEvents )
    return;
    
  sendWebsocket({
    'add_connection': {
      'input': drawflow.getNodeFromId(event['input_id'])['name'],
      'input_channel': Number(event['input_class'].split('_').at(-1)) - 1,
      'output': drawflow.getNodeFromId(event['output_id'])['name'],
      'output_channel': Number(event['output_class'].split('_').at(-1)) - 1
    } 
  });
}

function onNodeConnectionRemoved(event) {
  console.log(`onNodeConnectionRemoved(ignore=${ignoreGraphEvents})`, event);
  
  if( ignoreGraphEvents )
    return;
    
  sendWebsocket({
    'remove_connection': {
      'input': drawflow.getNodeFromId(event['input_id'])['name'],
      'input_channel': Number(event['input_class'].split('_').at(-1)) - 1,
      'output': drawflow.getNodeFromId(event['output_id'])['name'],
      'output_channel': Number(event['output_class'].split('_').at(-1)) - 1
    } 
  });
}

/*function onNodeContextMenu(event) {
  if( event.target.className != "drawflow" ) {
    console.log(`ignoring contextmenu event from '${event.srcElement.className}'`);
    return;
  }
  
  // return native menu if pressing control
  if( event.ctrlKey ) 
    return;
                
  console.log('onNodeContextMenu', event);
  //addNodeContextMenu(event.clientX, event.clientY);
  
  //$('#contextMenu').show().css({position: "absolute", left: event.clientX, top: event.clientY});
  //openContextMenu('#contextMenu', event);
  
  return false;
}*/

function addNodeContextMenu(x, y) {
  let html = `
    <ul class="dropdown-menu" id="node_context_menu" style="position: absolute; left: 0; top: 0;">
      <li><a class="dropdown-item">ABC</a></li>
    </ul>
  `;
   
  html = `
    <ul id="node_context_menu" style="position: absolute; left: ${x}; top: ${y};">
      <li><a>ABC</a></li>
    </ul>
  `;
  
  //html = `<p style="position: absolute; left: 100; top: 100;">ABC123</p>`;
   
  let contextMenu = fromHTML(html);
  //document.getElementById("drawflow").appendChild(contextMenu);
  document.body.appendChild(contextMenu);
  console.log('opening node editor context menu', contextMenu);
  return contextMenu;
}

function addDialog(id, title, html, xl, onsubmit, oncancel) {
  $('body').append(`
  <div class="modal fade" id="${id}" tabindex="-1" aria-labelledby="${id}_label" aria-hidden="true">
      <div class="modal-dialog ${xl ? 'modal-xl' : ''}">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="${id}_label">${title}</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close" style="color: #eeeeee;"></button>
          </div>
          <div class="modal-body">
            ${html}
          </div>
          <div class="modal-footer">
            <button id="${id}_cancel" type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button> 
            <button id="${id}_submit" type="button" class="btn btn-secondary" data-bs-dismiss="modal">Okay</button>
          </div>
        </div>
      </div>
    </div>
  `);
  
  if( onsubmit != undefined )
    $(`#${id}_submit`).on('click', onsubmit);
    
  if( oncancel != undefined )
    $(`#${id}_cancel`).on('click', oncancel);
}

function addPluginDialog(plugin_name, stage, title, description, parameters, max_per_column) {
  if( title == undefined )
    title = plugin_name;
    
  if( max_per_column == undefined )
    max_per_column = 6;
    
  const num_params = Object.keys(parameters).length;
  const num_columns = Math.ceil(num_params / max_per_column);
  
  const dialog_id = `${plugin_name}_${stage}_dialog`;
  const dialog_xl = (num_params > max_per_column);
  
  let html = '';
  
  if( description != null )
    html += `<p>${description}</p>`;
    
  html += `<div class="container">`;
  html += `<div class="row">`;
  html += `<div class="col-sm">`;

  var select2_args = {};
  let param_count = 0;
  
  for( param_name in parameters ) {
    const param = parameters[param_name];
    const id = `${plugin_name}_${stage}_${param_name}`;
    
    if( param['hidden'] )
      continue;
    
    if( param_count > 0 && param_count % Math.ceil(num_params / num_columns) == 0 )
      html += `</div><div class="col-sm">`;
      
    let value = '';
    let value_html = '';
    
    if( 'default' in param && param['default'] != undefined ) {
      value = param['default'];
      value_html = `value="${value}"`;
    }
     
    let help = '';
    
    if( 'help' in param )
      help = `<div id="${id}_help" class="form-text">${param['help']}</div>`;
    
    if( param['type'] == 'boolean' ) {
      param['options'] = [true, false]
    }
    
    const has_options = ('options' in param);
    const has_suggestions = ('suggestions' in param);
    
    if( has_options || has_suggestions ) {
      var input_html = '';
      var multiple_html = param.multiple ? 'multiple="multiple"' : '';
      
      if( has_options ) {
        var options = param['options'];
        if( options.length > 8 )
          select2_args[id] = {};
        else
          select2_args[id] = {minimumResultsForSearch: Infinity};
      }
      else if( has_suggestions ) {
        var options = param['suggestions'];
        select2_args[id] = {tags: true, placeholder: 'enter'}; //tags: true, placeholder: 'enter'};
      }
      
      input_html += `<br/><select id="${id}" class="form-control select2" style="width: 100%;" ${multiple_html}>\n`
      
      for( i in options ) {
        if( options[i] == value )
          var selected = ` selected="selected"`;
        else
          var selected = '';
        
        input_html += `  <option${selected}>${options[i]}</option>\n`
      }
      
      input_html += `</select>\n`;
    }
    /*else if( 'suggestions' in param ) {
      const list_id = `${id}_list`;
      var input_html = `<input id="${id}" type="${type}" class="form-control" list="${list_id}"/>`;
      
      input_html += `<datalist id="${list_id}">`;
      
      for( i in param['suggestions'] ) {
        input_html += `<option>${param['suggestions'][i]}</option>`;
      }
      
      input_html += `</datalist>`; 
    }*/
    else if( 'multiline' in param ) {
      var input_html = `<textarea id="${id}" class="form-control" aria-describedby="${id}_help" rows=${param['multiline']}>${value}</textarea>`;
    }
    else if( 'color' in param ) {
      var input_html = `<input id="${id}" type="color" class="form-control" aria-describedby="${id}_help" ${value_html}/>`;
    }
    else {
      var type = (param['type'] == 'integer') ? 'number' : param['type'];
      
      if( param.password )
        type = 'password';
        
      var input_html = `<input id="${id}" type="${type}" class="form-control" aria-describedby="${id}_help" ${value_html}>`;
    }

    html += `
      <div class="mb-3">
        <label for="${id}" class="form-label">${param['display_name']}</label>
        ${input_html}
        ${help}
      </div>
    `;
    
    param_count += 1;
  }
  
  html += `</div></div></div>`;
  
  let onsubmit = function() {
    //console.log(`onsubmit(${plugin_name})`);
    let args = {};
    
    for( param_name in parameters ) {
      if( parameters[param_name]['hidden'] )
        continue;
      let value = document.getElementById(`${plugin_name}_${stage}_${param_name}`).value;
      if( value != undefined && value.length > 0 ) { // input.value are always strings
        const type = parameters[param_name]['type'];
        if( type == 'integer' || type == 'number' )
          value = Number(value);
        args[param_name] = value;
      }
    }
    
    console.log(`${stage}Plugin(${plugin_name}) =>`);
    console.log(args);

    args['name'] = plugin_name;
    args['layout_node'] = nodeLayoutInsertPos();
    
    let msg = {};
    
    if( stage == 'init' ) {
      args['type'] = plugin_name;
      msg['add_plugin'] = args;
    }
    else {
      args['name'] = plugin_name;
      msg[`${stage}_plugin`] = args;
    }

    sendWebsocket(msg);
  }

  addDialog(dialog_id, title, html, dialog_xl, onsubmit);
  
  for( select2_id in select2_args ) {
    select2_args[select2_id]['dropdownParent'] = $(`#${dialog_id}`);
    $(`#${select2_id}`).select2(select2_args[select2_id]);
  }
  
  return dialog_id;
}

function fitGridWidgetContents(el) {
  //let width = parseInt(el.getAttribute('gs-w')) || 0;
  //let height = parseInt(el.getAttribute('gs-h')) || 0;
  //console.log('grid widget resize', width, height, el.offsetWidth, el.offsetHeight);
  //let card_title = el.querySelector('.card-title');
  //console.log('grid widget resize', card_title, card_title.offsetHeight);
  //el.querySelector('.collapse').style.height = `${el.offsetHeight - card_title.offsetHeight - 50}px`;
  console.log('grid widget resize', el.offsetWidth, el.offsetHeight);
  
  let grid_rect = el.getBoundingClientRect();

  let card = el.querySelector('.card-body');
  let card_rect = card.getBoundingClientRect();
  
  let card_inner = el.querySelector('.collapse');
  let card_inner_rect = card_inner.getBoundingClientRect();
  
  console.log(`grid widget rect height=${grid_rect.height}  bottom=${grid_rect.bottom}`);
  console.log(`card widget rect height=${card_rect.height}  bottom=${card_rect.bottom}`);
  console.log(`card inner rect height=${card_inner_rect.height}  bottom=${card_inner_rect.bottom}`);
  
  //if( card_inner_rect.bottom > card_rect.bottom ) {
    //card_inner.style.height = `${card_inner_rect.height - (card_inner_rect.bottom - card_rect.bottom)}px`;
    //console.log('set height', card_inner.style.height);
  //}
  /*if( card_rect.height > grid_rect.height ) {
    //card.style.height = `${grid_rect.height}px`;
    card_inner.style.height = `${card_inner.offsetHeight - (card_rect.height - grid_rect.height)}px`;
  }*/
  //card.style.height = `${el.offsetHeight}px`;
  //el.querySelector('.collapse').style.height = `${el.offsetHeight-150}px`;
}


/**
 * https://stackoverflow.com/a/35385518
 * @param {String} HTML representing a single element.
 * @param {Boolean} flag representing whether or not to trim input whitespace, defaults to true.
 * @return {Element | HTMLCollection | null}
 */
function fromHTML(html, trim = true) {
  // Process the HTML string.
  html = trim ? html.trim() : html;
  if (!html) return null;

  // Then set up a new template element.
  const template = document.createElement('template');
  template.innerHTML = html;
  const result = template.content.children;

  // Then return either an HTMLElement or HTMLCollection,
  // based on whether the input HTML had one or more roots.
  if (result.length === 1) return result[0];
  return result;
}

